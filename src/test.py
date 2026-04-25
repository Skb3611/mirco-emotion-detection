import hsemotion, os
path = os.path.join(os.path.dirname(hsemotion.__file__), "facial_emotions.py")
with open(path) as f:
    print(f.read())