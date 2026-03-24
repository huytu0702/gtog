$python = "F:/KL/gtog-eval/.venv/Scripts/python.exe"
$root = "F:/KL/gtog-eval/medical_graphrag_project"

$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

$configs = @(
  "F:/KL/gtog-eval/medical_graphrag_project/eval_configs/eval_tog_chunk_03.yaml",
  "F:/KL/gtog-eval/medical_graphrag_project/eval_configs/eval_tog_chunk_04.yaml"
)

foreach ($config in $configs) {
  Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "`$env:PYTHONIOENCODING='utf-8'; `$env:PYTHONUTF8='1'; & '$python' -X utf8 -m graphrag eval --root '$root' --config '$config' --methods 'tog' --skip-evaluation --verbose"
  )
}
