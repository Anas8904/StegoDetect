import os
import base64
import zipfile
import io

def decode_test_images(directory):
    """
    Decodes images that have been encrypted/encoded using base64 or zip.
    If the files are already valid PNG images, they will be left intact.
    """
    if not os.path.exists(directory):
        return
        
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if not os.path.isfile(filepath):
            continue
            
        with open(filepath, 'rb') as f:
            content = f.read()
            
        # If it's already a valid PNG, we don't need to decrypt it
        if content.startswith(b'\x89PNG'):
            continue
            
        # Decrypt Base64
        if filename.startswith('b64_'):
            try:
                decoded = base64.b64decode(content)
                with open(filepath, 'wb') as f:
                    f.write(decoded)
                print(f"Successfully decrypted base64 image: {filename}")
            except Exception as e:
                print(f"Failed to decrypt base64 image {filename}: {e}")
                
        # Decrypt ZIP
        elif filename.startswith('zip_'):
            try:
                with zipfile.ZipFile(io.BytesIO(content)) as z:
                    name = z.namelist()[0]
                    img_data = z.read(name)
                    with open(filepath, 'wb') as f:
                        f.write(img_data)
                print(f"Successfully decrypted zip image: {filename}")
            except Exception as e:
                print(f"Failed to decrypt zip image {filename}: {e}")

if __name__ == '__main__':
    # Test decryption on the test dataset directly
    print("Running decryption on test dataset...")
    decode_test_images('../data/combined/test/lsb')
    print("Decryption complete.")
