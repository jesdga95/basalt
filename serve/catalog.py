"""모델 카탈로그: 한 폴더의 .basalt 팩을 이름과 경로로 나열한다.

`serve --catalog`(또는 config `server.catalog: true`)가 켜지면 서버는 이 목록의 모든 팩을
`/v1/models` 로 신고하고, 요청의 model 이름이 현재 팩과 다르면 그 팩으로 엔진을 다시 띄운다.

이름은 팩 파일 안의 meta `name` 을 쓴다 — 양자화판마다 다르게 내보내져 있어 파일 이름 없이도
구분되고, OpenAI 클라이언트가 쓰는 id 와 같다. 읽을 수 없는(내려받다 잘린) 팩은 조용히 빠진다.
"""
from __future__ import annotations

import json
from pathlib import Path


def scan(directory: str | Path) -> list[dict]:
    """[{"id": 이름, "path": 절대 경로}] — 팩 파일이 스스로 신고하는 이름 순으로 정렬한다."""
    from serve.engine.modelfile import ModelFile      # tools/ 를 sys.path 에 넣는 모듈을 거쳐 가져온다

    out: list[dict] = []
    taken: set[str] = set()
    for p in sorted(Path(directory).glob("*.basalt")):
        try:
            meta = json.loads(ModelFile(str(p)).read("meta"))
        except Exception:                     # 잘린 팩·다른 형식: 목록에 넣지 않는다
            continue
        model_id = str(meta.get("name") or "").strip() or p.stem
        if model_id in taken:                 # 이름이 겹치는 팩: 파일 이름을 덧붙여 구분한다
            model_id = f"{model_id} [{p.stem}]"
        taken.add(model_id)
        out.append({"id": model_id, "path": str(p.resolve())})
    out.sort(key=lambda e: e["id"].lower())
    return out


def find(catalog: list[dict], name: str) -> dict | None:
    """카탈로그에서 그 이름의 팩; 없으면 None."""
    return next((e for e in catalog if e["id"] == name), None)
