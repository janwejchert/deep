import requests
import struct
import io

url = "https://zenodo.org/api/records/13617673/files/pmcardio_ecg_image_database.zip/content"

r = requests.head(url)
size = int(r.headers.get('Content-Length', 0))

if size > 0:
    fetch_size = 1024 * 1024
    start = max(0, size - fetch_size)
    headers = {"Range": f"bytes={start}-{size-1}"}
    r = requests.get(url, headers=headers)
    data = r.content
    print(f"Fetched {len(data)} bytes from end of file.")
    
    zip64_locator_sig = b'PK\x06\x07'
    idx = data.rfind(zip64_locator_sig)
    if idx != -1:
        print("Found ZIP64 EOCD Locator!")
        locator = data[idx:idx+20]
        _, disk_eocd, eocd64_offset, total_disks = struct.unpack('<IIQI', locator)
        print(f"ZIP64 EOCD Offset: {eocd64_offset}")
        
        if eocd64_offset >= start:
            eocd64 = data[eocd64_offset - start : eocd64_offset - start + 56]
        else:
            r_eocd64 = requests.get(url, headers={"Range": f"bytes={eocd64_offset}-{eocd64_offset+55}"})
            eocd64 = r_eocd64.content
            
        _, _, _, _, _, _, num_entries_disk, num_entries, cd_size, cd_offset = struct.unpack('<IQHHIIQQQQ', eocd64[:56])
        print(f"ZIP64 CD size: {cd_size}, CD offset: {cd_offset}, entries: {num_entries}")
        
        print("Fetching Central Directory...")
        r_cd = requests.get(url, headers={"Range": f"bytes={cd_offset}-{cd_offset+cd_size-1}"})
        cd_data = r_cd.content
        
        offset = 0
        metadata_entry = None
        for i in range(num_entries):
            if offset >= len(cd_data):
                break
            sig = cd_data[offset:offset+4]
            if sig != b'PK\x01\x02':
                break
            header = cd_data[offset:offset+46]
            # Fixed format: 6 H, 3 I, 5 H, 2 I = 42 bytes
            fields = struct.unpack('<HHHHHHIIIHHHHHII', header[4:46])
            
            comp_size = fields[7]
            uncomp_size = fields[8]
            fn_len = fields[9]
            extra_len = fields[10]
            comment_len = fields[11]
            local_offset = fields[15]
            
            fn = cd_data[offset+46:offset+46+fn_len].decode('utf-8', errors='ignore')
            
            if local_offset == 0xFFFFFFFF or comp_size == 0xFFFFFFFF:
                extra_data = cd_data[offset+46+fn_len : offset+46+fn_len+extra_len]
                ext_offset = 0
                while ext_offset < len(extra_data):
                    ex_id, ex_size = struct.unpack('<HH', extra_data[ext_offset:ext_offset+4])
                    ext_offset += 4
                    if ex_id == 0x0001:
                        ptr = 0
                        if fields[8] == 0xFFFFFFFF: # uncomp_size
                            uncomp_size = struct.unpack('<Q', extra_data[ext_offset+ptr:ext_offset+ptr+8])[0]
                            ptr += 8
                        if fields[7] == 0xFFFFFFFF: # comp_size
                            comp_size = struct.unpack('<Q', extra_data[ext_offset+ptr:ext_offset+ptr+8])[0]
                            ptr += 8
                        if local_offset == 0xFFFFFFFF:
                            local_offset = struct.unpack('<Q', extra_data[ext_offset+ptr:ext_offset+ptr+8])[0]
                            ptr += 8
                    ext_offset += ex_size

            if 'metadata.csv' in fn.lower():
                print(f"Found metadata.csv! Local offset: {local_offset}, compressed size: {comp_size}")
                metadata_entry = {
                    'name': fn,
                    'offset': local_offset,
                    'comp_size': comp_size,
                    'compression': fields[3]
                }
                break
                
            offset += 46 + fn_len + extra_len + comment_len
            
        if metadata_entry:
            fetch_start = metadata_entry['offset']
            fetch_end = fetch_start + metadata_entry['comp_size'] + 2000
            print(f"Fetching file data from {fetch_start} to {fetch_end}...")
            r_file = requests.get(url, headers={"Range": f"bytes={fetch_start}-{fetch_end}"})
            file_data = r_file.content
            
            if file_data[:4] == b'PK\x03\x04':
                local_fields = struct.unpack('<HHHHHIIIHH', file_data[4:30])
                l_comp_size = local_fields[6]
                l_fn_len = local_fields[8]
                l_extra_len = local_fields[9]
                
                if l_comp_size == 0xFFFFFFFF:
                    l_comp_size = metadata_entry['comp_size']
                    
                data_start = 30 + l_fn_len + l_extra_len
                actual_data = file_data[data_start:data_start + l_comp_size]
                
                with open('/Users/felipedeleon/Desktop/Deep Ler,Project/metadata.csv', 'wb') as f:
                    f.write(actual_data)
                print("Successfully extracted metadata.csv!")
                
                if metadata_entry['compression'] == 8:
                    import zlib
                    decompressed = zlib.decompress(actual_data, -15)
                    with open('/Users/felipedeleon/Desktop/Deep Ler,Project/metadata.csv', 'wb') as f:
                        f.write(decompressed)
                    print("Successfully decompressed metadata.csv!")
            else:
                print("Local header signature not found.")
    else:
        print("ZIP64 EOCD Locator not found in last 1MB")
else:
    print("Could not determine file size.")
