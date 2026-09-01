# Building MkDocs Document Locally

The following describes how to build the MkDocs documentation locally for real-time preview and debugging during documentation writing.

## Installing Dependencies

```shell
pip install -r requirements/mkdocs.txt
```

## Starting the Local Service

Step 1: Run the following command in the root directory of the project to start the local MkDocs service:

```shell
mkdocs serve
```

Step 2: After the service is started, the terminal displays information similar to the following:

```text
INFO     -  Building documentation...
INFO     -  Cleaning site directory
INFO     -  Documentation built in 1.23 s
INFO     -  [12:00:00] Watching paths for changes
INFO     -  [12:00:00] Serving on http://127.0.0.1:8000/
```

Step 3: Access `http://127.0.0.1:8000/` in the browser to preview the document.

> [!NOTE]
> By default, `mkdocs serve` listens on port `8000`. If the port is occupied, you can use the `-a` option to specify another port, for example, `mkdocs serve -a 127.0.0.1:8080`.

## FAQs

### Dependency Installation Failure

If a dependency conflict occurs or the installation fails during the `pip install` process, you are advised to use a virtual environment for isolation.

```shell
python -m venv .venv_mkdocs
source .venv_mkdocs/bin/activate
pip install -r requirements/mkdocs.txt
```
