import sys
from pathlib import Path
import math

CHUNK_SIZE_LINES = 300 # Limit chunk size to prevent token overflow

def sanitize_and_chunk_log(log_path: Path):
    if not log_path.exists():
        print(f"Error: Log file {log_path} does not exist.")
        sys.exit(1)
        
    try:
        content = log_path.read_text(encoding="utf-8").splitlines()
    except Exception as e:
        print(f"Error reading log file: {e}")
        sys.exit(1)
        
    if not content:
        return
        
    total_lines = len(content)
    num_chunks = math.ceil(total_lines / CHUNK_SIZE_LINES)
    
    # Write chunks
    for i in range(num_chunks):
        start_idx = i * CHUNK_SIZE_LINES
        end_idx = min(start_idx + CHUNK_SIZE_LINES, total_lines)
        chunk_content = "\n".join(content[start_idx:end_idx])
        
        chunk_file = log_path.with_name(f"{log_path.stem}_chunk_{i+1}{log_path.suffix}")
        chunk_file.write_text(chunk_content, encoding="utf-8")
        
    # We rename the original log to avoid reprocessing
    archive_path = log_path.with_name(f"{log_path.stem}_full.archive")
    log_path.rename(archive_path)
    print(f"Log sanitized and chunked into {num_chunks} files.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python Log_Sanitizer.py <path_to_log_file>")
        sys.exit(1)
        
    target_log = Path(sys.argv[1]).resolve()
    sanitize_and_chunk_log(target_log)
