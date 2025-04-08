# MCP COMANDES

### Instal·lar dependències
```
pip install mcp uv
```

### Activar entorn UV
```
uv init
uv add "mcp[cli]"
```

### Executar MCP Inspector
```
mcp dev main.py
```
- Hem d'activar l'entorn virtual que es crea al fer l' ```uv init```

### Afegir tool al 5ire app
```
uv run --with mcp[cli] mcp run /home/iticbcn/Escritorio/UA/Model_Context_Protocol/main.py