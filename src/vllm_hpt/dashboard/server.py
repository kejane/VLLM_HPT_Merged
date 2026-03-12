import asyncio
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from vllm_hpt.dashboard.state import get_dashboard_state

app = FastAPI(title="vLLM HPT Dashboard")

STATIC_DIR = Path(__file__).parent / "static"


class StartEvaluateRequest(BaseModel):
    params: dict


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/evaluate")
async def evaluate_page():
    return FileResponse(STATIC_DIR / "evaluate.html")


@app.post("/api/start-evaluate")
async def start_evaluate(request: StartEvaluateRequest):
    state = get_dashboard_state()
    state.submit_params(request.params)
    return {"status": "ok", "params": request.params}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    state = get_dashboard_state()
    await state.subscribe(websocket)
    
    try:
        await websocket.send_text(__import__("json").dumps(state.to_dict()))
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await state.unsubscribe(websocket)


@app.get("/api/optuna/history")
async def optuna_history():
    state = get_dashboard_state()
    if state.optuna_study is None:
        return {"trials": [], "message": "No active study"}

    trials = []
    for trial in state.optuna_study.trials:
        if trial.state.name == "COMPLETE":
            trials.append({
                "number": trial.number,
                "value": trial.value,
                "params": trial.params,
                "state": trial.state.name,
            })
    return {"trials": trials}


@app.get("/api/optuna/importance")
async def optuna_importance():
    state = get_dashboard_state()
    if state.optuna_study is None:
        return {"importances": {}, "message": "No active study"}

    import optuna
    try:
        completed = [t for t in state.optuna_study.trials
                     if t.state == optuna.trial.TrialState.COMPLETE]
        if len(completed) < 2:
            return {"importances": {}, "message": "Need at least 2 completed trials"}

        importances = optuna.importance.get_param_importances(state.optuna_study)
        return {"importances": importances}
    except Exception as e:
        return {"importances": {}, "message": str(e)}


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def run_server(host: str = "127.0.0.1", port: int = 8501):
    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level="warning")


async def run_server_background(host: str = "127.0.0.1", port: int = 8501):
    import uvicorn
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    await server.serve()
