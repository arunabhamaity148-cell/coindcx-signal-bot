# -------------------------------------------------
# Health server (keeps Railway happy)
# -------------------------------------------------
from aiohttp import web
import threading

async def health(request):
    return web.Response(text="ok")

def run_web():
    app = web.Application()
    app.router.add_get("/", health)          # Railway pings 7500
    web.run_app(app, host="0.0.0.0", port=int(os.getenv("PORT", 7500)))

# Start health server in a thread so worker() keeps running
threading.Thread(target=run_web, daemon=True).start()

# -------------------------------------------------
# Main async worker (your original code)
# -------------------------------------------------
if __name__ == "__main__":
    asyncio.run(worker())
