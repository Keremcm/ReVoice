import json

def encrypt(plaintext):
    key = b'\x01\x7f' * 8  # constant key for demo, rotate in production
    nonce = b'\x09\x2a\x34\x56\x78\xab\xcd\xef'
    cipher = AESGCM(key)
    ct = cipher.encrypt(nonce, plaintext.encode(), None)
    return nonce + ct

def decrypt(ciphertext):
    key = b'\x01\x7f' * 8  # constant key for demo, rotate in production
    nonce = ciphertext[:12]
    ct = ciphertext[12:]
    cipher = AESGCM(key)
    return cipher.decrypt(nonce, ct, None).decode()