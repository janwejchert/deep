import zipfile
import struct
import os

zip_path = '/Users/felipedeleon/Desktop/Deep Ler,Project/data images /13617673.zip'

def list_zip_entries(zip_fn):
    print(f"Scanning local file headers in zip: {zip_fn}")
    with open(zip_fn, 'rb') as f:
        # Check first signature
        sig = f.read(4)
        if sig != b'PK\x03\x04':
            print("Not a valid ZIP signature.")
            return
            
        f.seek(0)
        # We want to find all file names
        count = 0
        while True:
            pos = f.tell()
            sig = f.read(4)
            if len(sig) < 4:
                break
            if sig != b'PK\x03\x04':
                # Try searching for next PK\x03\x04 in case of trash
                # But typically local headers are contiguous
                break
            
            # Read header
            header = f.read(26)
            if len(header) < 26:
                break
            version, flags, compression, mod_time, mod_date, crc, comp_size, uncomp_size, fn_len, extra_len = struct.unpack('<HHHHHIIIHH', header)
            fn_bytes = f.read(fn_len)
            fn = fn_bytes.decode('utf-8', errors='ignore')
            f.seek(extra_len, 1)
            
            # Print if it looks like metadata.csv or is not a jpeg
            if 'metadata' in fn.lower() or not fn.lower().endswith('.jpeg'):
                print(f"Header at {pos}: {fn} (comp_size: {comp_size})")
                
            f.seek(comp_size, 1)
            count += 1
            if count % 1000 == 0:
                print(f"Scanned {count} headers...")
                
    print(f"Scan complete. Total entries: {count}")

list_zip_entries(zip_path)
