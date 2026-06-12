import struct
import os
import zlib

def extract_nested_truncated(outer_zip, extract_dir):
    os.makedirs(extract_dir, exist_ok=True)
    with open(outer_zip, 'rb') as f:
        # Find the start of the first entry in the outer zip
        sig = f.read(4)
        if sig != b'PK\x03\x04':
            print("Outer file is not a valid ZIP.")
            return
        
        # Read local header of the outer zip
        header = f.read(26)
        version, flags, compression, mod_time, mod_date, crc, compressed_size, uncompressed_size, filename_len, extra_len = struct.unpack('<HHHHHIIIHH', header)
        filename = f.read(filename_len).decode('utf-8', errors='ignore')
        extra = f.read(extra_len)
        
        print(f"Outer entry: {filename}")
        
        # Now the file pointer f is exactly at the start of the inner zip data!
        # Let's read the inner zip from this position
        count = 0
        while True:
            inner_sig = f.read(4)
            if len(inner_sig) < 4:
                print("End of file reached.")
                break
            if inner_sig != b'PK\x03\x04':
                # If we don't see the PK header, we might be at the end of the valid data stream
                print(f"Non-ZIP signature found: {inner_sig}, stopping.")
                break
                
            # Read local file header of the inner file
            header = f.read(26)
            if len(header) < 26:
                print("Truncated header, stopping.")
                break
            version, flags, compression, mod_time, mod_date, crc, compressed_size, uncompressed_size, filename_len, extra_len = struct.unpack('<HHHHHIIIHH', header)
            filename = f.read(filename_len).decode('utf-8', errors='ignore')
            extra = f.read(extra_len)
            
            # Read compressed data
            data = f.read(compressed_size)
            if len(data) < compressed_size:
                print(f"Truncated data for {filename}, stopping.")
                break
                
            # If it's a directory, create it
            if filename.endswith('/'):
                os.makedirs(os.path.join(extract_dir, filename), exist_ok=True)
                continue
                
            # Extract file
            out_path = os.path.join(extract_dir, filename)
            out_dir = os.path.dirname(out_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
                
            if compression == 0: # Store
                with open(out_path, 'wb') as out_f:
                    out_f.write(data)
                count += 1
            elif compression == 8: # Deflate
                try:
                    decompressed = zlib.decompress(data, -15)
                    with open(out_path, 'wb') as out_f:
                        out_f.write(decompressed)
                    count += 1
                except Exception as e:
                    print(f"Failed to decompress {filename}: {e}")
            
            if count % 100 == 0:
                print(f"Extracted {count} files...")
                
        print(f"Extraction complete! Total files extracted: {count}")

if __name__ == '__main__':
    extract_nested_truncated(
        '/Users/felipedeleon/Desktop/Deep Ler,Project/data images /13617673.zip',
        '/Users/felipedeleon/Desktop/Deep Ler,Project/dataset_new_extracted'
    )
