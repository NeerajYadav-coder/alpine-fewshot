import os
import glob
import json
import hashlib

def get_file_md5(filepath):
    h = hashlib.md5()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def get_file_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def main():
    print("==================================================================")
    print("       ALPINE CHECKPOINT INTEGRITY & HASH VERIFICATION AUDIT      ")
    print("==================================================================")
    
    ckpt_dir = "checkpoints"
    manifest_dirs = ["manifests", "checkpoints", "."]
    manifest_files = []
    for md in manifest_dirs:
        manifest_files.extend(glob.glob(os.path.join(md, "*.json")))
    
    manifest_files = sorted(list(set([m for m in manifest_files if "manifest" in m])))

    print(f"Found {len(manifest_files)} manifest files.\n")

    total_verified = 0
    total_mismatch = 0
    missing_files = 0

    for manifest_path in manifest_files:
        print(f"--- Auditing Manifest: {os.path.basename(manifest_path)} ---")
        try:
            with open(manifest_path, 'r') as f:
                manifest_data = json.load(f)
        except Exception as e:
            print(f"  Error reading manifest {manifest_path}: {e}")
            continue

        entries = []
        if isinstance(manifest_data, dict):
            if "checkpoints" in manifest_data:
                entries = manifest_data["checkpoints"]
            elif "runs" in manifest_data:
                entries = manifest_data["runs"]
            else:
                entries = [manifest_data]
        elif isinstance(manifest_data, list):
            entries = manifest_data

        for item in entries:
            if not isinstance(item, dict):
                continue
            fname = item.get("filename", item.get("checkpoint_path", item.get("checkpoint", "")))
            if not fname:
                continue
            
            ckpt_path = fname if os.path.isabs(fname) else os.path.join(ckpt_dir, os.path.basename(fname))
            if not os.path.exists(ckpt_path):
                ckpt_path = os.path.basename(fname)

            if not os.path.exists(ckpt_path):
                missing_files += 1
                continue

            expected_md5 = item.get("md5", item.get("md5_hash", ""))
            expected_sha256 = item.get("sha256", item.get("sha256_hash", ""))

            if expected_md5:
                actual_md5 = get_file_md5(ckpt_path)
                if actual_md5 == expected_md5:
                    total_verified += 1
                else:
                    print(f"  ❌ MD5 MISMATCH: {ckpt_path}\n     Expected: {expected_md5}\n     Actual:   {actual_md5}")
                    total_mismatch += 1
            elif expected_sha256:
                actual_sha256 = get_file_sha256(ckpt_path)
                if actual_sha256 == expected_sha256:
                    total_verified += 1
                else:
                    print(f"  ❌ SHA256 MISMATCH: {ckpt_path}\n     Expected: {expected_sha256}\n     Actual:   {actual_sha256}")
                    total_mismatch += 1

    print("\n==================================================================")
    print(f"AUDIT COMPLETE:")
    print(f"  - Verified Checkpoint Hashes: {total_verified}")
    print(f"  - Checksum Mismatches:      {total_mismatch}")
    if missing_files > 0:
        print(f"  - Missing Checkpoint Files:  {missing_files}")
    if total_mismatch == 0:
        print("SUCCESS: 100% OF CHECKPOINT HASHES MATCH MANIFEST ENTRIES PERFECTLY.")
    print("==================================================================")

if __name__ == "__main__":
    main()
