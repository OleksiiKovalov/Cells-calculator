"""Runtime fuzzing harness for the NEW Cells Calculator pipeline.

Feeds generated / edge-case / corpus-mutated images through
``read_img`` + ``Model.inference`` and checks output invariants. Reimplemented
against NEW's ``Model.inference(ndarray) -> DataFrame`` seam (OLD's harness
targeted the removed ``{Cells, Nuclei, %}`` contract).

CLI:
    python -m tests.fuzzing --list-models
    python -m tests.fuzzing --model "YOLO-512 Segmenter" --max-cases 200 \
        --profile mixed --seed 0 --corpus testimages
"""
