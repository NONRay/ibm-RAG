import os
import json
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional, List, Union, Literal
from pydantic import BaseModel, Field
from openai import OpenAI
from src.api_requests import BaseOpenaiProcessor, AsyncOpenaiProcessor
import tiktoken
from tqdm import tqdm
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import time
import pandas as pd
from io import StringIO

message_queue = Queue()

class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            message_queue.put((record.levelno, msg))
        except Exception:
            self.handleError(record)

def process_messages():
    while not message_queue.empty():
        level, msg = message_queue.get_nowait()
        tqdm.write(msg)

class TableSerializer(BaseOpenaiProcessor):
    def __init__(self, preserve_temp_files: bool = True):
        super().__init__()
        self.preserve_temp_files = preserve_temp_files
        os.makedirs('./temp', exist_ok=True)
        
        self.logger = logging.getLogger('TableSerializer')
        self.logger.setLevel(logging.INFO)
        
        self.logger.handlers.clear()
        
        handler = TqdmLoggingHandler()
        handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        self.logger.addHandler(handler)
        
        self.logger.propagate = False

    def _chunk_table_html(self, table_html: str, max_rows: int = 40) -> List[str]:
        """Split a large HTML table into smaller chunks while preserving headers."""
        try:
            # Check if table_html is valid HTML
            if not table_html or not isinstance(table_html, str):
                return [table_html]
            
            # Use lxml parser for better compatibility
            try:
                dfs = pd.read_html(StringIO(table_html), flavor='lxml')
            except ImportError:
                 # Fallback to bs4/html5lib if lxml is missing
                dfs = pd.read_html(StringIO(table_html))
            except Exception:
                # If read_html fails completely (e.g. malformed HTML), return original
                return [table_html]

            if not dfs:
                return [table_html]
            
            df = dfs[0]
            if len(df) <= max_rows:
                return [table_html]
            
            chunks = []
            # Iterate through the dataframe in chunks
            for i in range(0, len(df), max_rows):
                chunk_df = df.iloc[i:i+max_rows]
                # Convert chunk back to HTML
                chunk_html = chunk_df.to_html(index=False, border=0, escape=False) # escape=False to preserve inner HTML if any
                chunks.append(chunk_html)
            
            self.logger.info(f"Split table into {len(chunks)} chunks (original rows: {len(df)})")
            return chunks
            
        except Exception as e:
            self.logger.warning(f"Failed to chunk table: {str(e)}. Using original table.")
            return [table_html]

    def _get_table_context(self, json_report, target_table_index):
        table_info = next(
            table
            for table in json_report.get("tables", [])
            if isinstance(table, dict) and table.get("table_id") == target_table_index
        )
        page_num = table_info.get("page")

        page_entry = next(
            (
                page
                for page in json_report.get("content", [])
                if isinstance(page, dict) and page.get("page") == page_num
            ),
            None,
        )

        page_content = []
        if page_entry and isinstance(page_entry.get("content"), list):
            page_content = page_entry["content"]

        if not page_content:
            self.logger.warning(f"Page {page_num} not found for table {target_table_index}")
            return "", ""

        # Find position of target table in page_content
        current_table_position = -1
        for i, block in enumerate(page_content):
            if not isinstance(block, dict):
                continue
            if block.get("type") == "table" and block.get("table_id") == target_table_index:
                current_table_position = i
                break

        # Find position of previous table if exists
        previous_table_position = -1
        for i in range(current_table_position-1, -1, -1):
            block = page_content[i]
            if isinstance(block, dict) and block.get("type") == "table":
                previous_table_position = i
                break

        # Find position of next table if exists
        next_table_position = -1
        for i in range(current_table_position + 1, len(page_content)):
            block = page_content[i]
            if isinstance(block, dict) and block.get("type") == "table":
                next_table_position = i
                break

        # Get blocks above current table
        start_position = previous_table_position + 1 if previous_table_position != -1 else 0
        context_before = page_content[start_position:current_table_position]

        # Get blocks after current table
        context_after = []
        if next_table_position == -1:
            # If no next table, take up to 3 blocks until end of page
            context_after = page_content[current_table_position + 1:current_table_position + 4]
        else:
            # If next table exists, take up to 3 blocks before it
            blocks_between = next_table_position - (current_table_position + 1)
            if blocks_between > 3:
                context_after = page_content[current_table_position + 1:current_table_position + 4]
            elif blocks_between > 1:
                context_after = page_content[current_table_position + 1:current_table_position + blocks_between]

        context_before_texts = []
        for block in context_before:
            if isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    context_before_texts.append(text)

        context_after_texts = []
        for block in context_after:
            if isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    context_after_texts.append(text)

        context_before = "\n".join(context_before_texts)
        context_after = "\n".join(context_after_texts)

        return context_before, context_after

    def _send_serialization_request(self, table, context_before, context_after):
        user_prompt = ""
        
        if context_before:
            user_prompt += f'Here is additional text before the table that might be relevant (or not):\n"""{context_before}"""\n\n'
        
        user_prompt += f'Here is a table in HTML format:\n"""{table}"""'
        
        if context_after:
            user_prompt += f'\n\nHere is additional text after the table that might be relevant (or not):\n"""{context_after}"""'
        
        system_prompt = TableSerialization.system_prompt
        reponse_schema = TableSerialization.TableBlocksCollection

        answer_dict = self.send_message(
            model='deepseek-chat',
            temperature=0,
            system_content=system_prompt,
            human_content=user_prompt,
            is_structured=True,
            response_format=reponse_schema
        )

        input_message = user_prompt + system_prompt + str(reponse_schema.schema())
        input_tokens = self.count_tokens(input_message)
        output_tokens = self.count_tokens(str(answer_dict))

        result = answer_dict
        return result
    
    def _serialize_table(self, json_report: dict, target_table_index: int) -> dict:

        # Get the context surrounding the table
        context_before, context_after = self._get_table_context(json_report, target_table_index)
        
        # Get the table content
        table_info = next(table for table in json_report["tables"] if table["table_id"] == target_table_index)
        table_content = table_info["html"]
        
        # Serialize the table with its context
        result = self._send_serialization_request(
            table=table_content,
            context_before=context_before,
            context_after=context_after
        )
        
        return result

    def serialize_tables(self, json_report: dict) -> dict:
        """Process all tables in the report and add serialization results to each table's info"""
        
        for table in json_report["tables"]:
            table_index = table["table_id"]
            
            # Get serialization results for current table
            serialization_result = self._serialize_table(
                json_report=json_report,
                target_table_index=table_index
            )
            
            # Add serialization results to the table info
            table["serialized"] = serialization_result
        
        return json_report

    async def async_serialize_tables(
        self, 
        json_report: dict,
        requests_filepath: str = './temp_async_llm_requests.jsonl',
        results_filepath: str = './temp_async_llm_results.jsonl'
    ) -> dict:
        """Process all tables in the report asynchronously"""
        queries = []
        # Mapping from table_index to list of query indices
        chunk_map = {} 
        query_idx = 0
        
        for table in json_report["tables"]:
            table_index = table["table_id"]
            if table_index not in chunk_map:
                chunk_map[table_index] = []
            
            context_before, context_after = self._get_table_context(json_report, table_index)
            table_info = next(table for table in json_report["tables"] if table["table_id"] == table_index)
            table_content = table_info["html"]
            
            # Split table if it's too large
            table_chunks = self._chunk_table_html(table_content)
            
            for i, chunk_html in enumerate(table_chunks):
                # Construct the query for this chunk
                query = ""
                if context_before:
                    query += f'Here is additional text before the table that might be relevant (or not):\n"""{context_before}"""\n\n'
                
                # If chunked, add a small hint
                if len(table_chunks) > 1:
                    query += f'Here is part {i+1} of {len(table_chunks)} of a table in HTML format:\n"""{chunk_html}"""'
                else:
                    query += f'Here is a table in HTML format:\n"""{chunk_html}"""'
                    
                if context_after:
                    query += f'\n\nHere is additional text after the table that might be relevant (or not):\n"""{context_after}"""'
                
                queries.append(query)
                chunk_map[table_index].append(query_idx)
                query_idx += 1

        base_url = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1").rstrip('/')
        request_url = f"{base_url}/chat/completions"
        api_key = os.getenv("QWEN_API_KEY")

        results = await AsyncOpenaiProcessor().process_structured_ouputs_requests(
            model='qwen-plus',
            temperature=0,
            system_content=TableSerialization.system_prompt,
            queries=queries,
            response_format=TableSerialization.TableBlocksCollection,
            preserve_requests=False,
            preserve_results=False,
            logging_level=20,
            requests_filepath=requests_filepath,
            save_filepath=results_filepath,
            request_url=request_url,
            api_key=api_key,
        )

        # Add results back to json_report
        for table_index, q_indices in chunk_map.items():
            # Gather all results for this table
            
            merged_result = {
                "subject_core_entities_list": [],
                "relevant_headers_list": [],
                "information_blocks": []
            }
            
            has_valid_result = False
            
            for q_i in q_indices:
                if q_i < len(results):
                    res = results[q_i]
                    ans = res.get('answer')
                    if ans and isinstance(ans, dict):
                        has_valid_result = True
                        merged_result["subject_core_entities_list"].extend(ans.get("subject_core_entities_list", []))
                        merged_result["information_blocks"].extend(ans.get("information_blocks", []))
                        # For headers, we take the first valid one, or merge them?
                        # Usually headers are same for all chunks.
                        if not merged_result["relevant_headers_list"]:
                            merged_result["relevant_headers_list"] = ans.get("relevant_headers_list", [])
            
            if has_valid_result:
                table_info = next(table for table in json_report["tables"] if table["table_id"] == table_index)
                
                new_table = {}
                for key, value in table_info.items():
                    new_table[key] = value
                    if key == "html":
                        new_table["serialized"] = merged_result
                
                for i, table in enumerate(json_report["tables"]):
                    if table["table_id"] == table_index:
                        json_report["tables"][i] = new_table

        return json_report

    def process_file(self, json_path: Path) -> None:
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                json_report = json.load(f)
            
            thread_id = threading.get_ident()
            requests_filepath = f'./temp/async_llm_requests_{thread_id}.jsonl'
            results_filepath = f'./temp/async_llm_results_{thread_id}.jsonl'
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                updated_report = loop.run_until_complete(self.async_serialize_tables(
                    json_report,
                    requests_filepath=requests_filepath,
                    results_filepath=results_filepath
                ))
            finally:
                loop.close()
                try:
                    os.remove(requests_filepath)
                    os.remove(results_filepath)
                except FileNotFoundError:
                    pass
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(updated_report, f, indent=2, ensure_ascii=False)
                
        except json.JSONDecodeError as e:
            self.logger.error("JSON Error in %s: %s", json_path.name, str(e))
            raise
        except Exception as e:
            self.logger.error("Error processing %s: %s", json_path.name, str(e))
            raise

    def process_directory_parallel(self, input_dir: Path, max_workers: int = 5):
        """Process JSON files in parallel using thread pool.
        
        Args:
            input_dir: Path to directory containing JSON files
            max_workers: Maximum number of threads to use
        """
        self.logger.info("Starting parallel table serialization...")
        
        json_files = list(input_dir.glob("*.json"))
        
        if not json_files:
            self.logger.warning("No JSON files found in %s", input_dir)
            return

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            with tqdm(
                total=len(json_files),
                desc="Processing files",
                mininterval=1.0,
                maxinterval=5.0,
                smoothing=0.3
            ) as pbar:
                futures = []
                for json_file in json_files:
                    future = executor.submit(self.process_file, json_file)
                    future.add_done_callback(lambda p: pbar.update(1))
                    futures.append(future)
                
                while futures:
                    process_messages()
                    
                    done_futures = []
                    for future in futures:
                        if future.done():
                            done_futures.append(future)
                            try:
                                future.result()
                            except Exception as e:
                                self.logger.error(str(e))
                    
                    for future in done_futures:
                        futures.remove(future)
                    
                    time.sleep(0.1)

        process_messages()
        self.logger.info("Table serialization completed!")


class TableSerialization:
        
    system_prompt = (
        "You are a table serialization agent.\n"
        "Your task is to create a set of contextually independent blocks of information based on the provided table and surrounding text.\n"
        "These blocks must be totally context-independent because they will be used as separate chunk to populate database."
    )

    class SerializedInformationBlock(BaseModel):
        "A single self-contained information block enriched with comprehensive context"

        subject_core_entity: str = Field(description="A primary focus of what this block is about. Usually located in a row header. If one row in the table doesn't make sense without neighboring rows, you can merge information from neighboring rows into one block")
        information_block: str = Field(description=(
    "Detailed information about the chosen core subject from tables and additional texts. Information SHOULD include:\n"
    "1. All related header information\n"
    "2. All related units and their descriptions\n"
    "    2.1. If header is Total, always write additional context about what this total represents in this block!\n"
    "3. All additional info for context enrichment to make ensure complete context-independency if it present in whole table. This can include:\n"
    "    - The name of the table\n"
    "    - Additional footnotes\n"
    "    - The currency used\n"
    "    - The way amounts are presented\n"
    "    - Anything else that can make context even slightly richer\n"
    "SKIPPING ANY VALUABLE INFORMATION WILL BE HEAVILY PENALIZED!"
    ))

    class TableBlocksCollection(BaseModel):
        """Collection of serialized table blocks with their core entities and header relationships"""

        subject_core_entities_list: List[str] = Field(
            description="A complete list of core entities. Keep in mind, empty headers are possible - they should also be interpreted and listed (Usually it's a total or something similar). In most cases each row header represents a core entity")
        relevant_headers_list: List[str] = Field(description="A list of ALL headers relevant to the subject. These headers will serve as keys in each information block. In most cases each column header represents a core entity")
        information_blocks: List["TableSerialization.SerializedInformationBlock"] = Field(description="Complete list of fully described context-independent information blocks")
