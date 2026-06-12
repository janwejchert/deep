import struct
import os

zip_path = '/Users/felipedeleon/Desktop/Deep Ler,Project/data images /13617673.zip'

def scan_inner_zip(zip_fn):
    with open(zip_fn, 'rb') as f:
        # Seek past the outer zip local header
        # Outer header:
        # sig (4) + header (26) + fn_len + extra_len
        f.read(4)
        header = f.read(26)
        version, flags, compression, mod_time, mod_date, crc, comp_size, uncomp_size, fn_len, extra_len = struct.unpack('<HHHHHIIIHH', header)
        f.seek(fn_len + extra_len, 1)
        
        # Now we are at the start of the inner zip
        print(f"Inner zip starts at tell(): {f.tell()}")
        
        count = 0
        while True:
            pos = f.tell()
            sig = f.read(4)
            if len(sig) < 4:
                print("EOF reached.")
                break
            if sig != b'PK\x03\x04':
                print(f"Non-local header signature {sig} at {pos}, stopping.")
                break
                
            header = f.read(26)
            if len(header) < 26:
                print("Truncated header, stopping.")
                break
            version, flags, compression, mod_time, mod_date, crc, compressed_size, uncompressed_size, filename_len, extra_len = struct.unpack('<HHHHHIIIHH', header)
            filename = f.read(filename_len).decode('utf-8', errors='ignore')
            f.seek(extra_len, 1)
            
            # Print entries that don't end with .jpeg or contain metadata
            if 'metadata' in filename.lower() or not filename.lower().endswith('.jpeg'):
                print(f"Inner Entry at {pos}: {filename} (comp_size: {compressed_size}, uncomp_size: {uncompressed_size})")
                
            f.seek(compressed_size, 1)
            count += 1
            if count % 200 == 0:
                print(f"Scanned {count} inner entries...")
                
    print(f"Total inner entries scanned: {count}")

scan_inner_zip(zip_path)
