# Deleting Files Without Owners

Files without owners are not allowed in the system. If such a file is found, you need to delete it or change the file owner.

Search for files without owners:

```bash
find / -nouser -print
find / -nogroup -print
```
