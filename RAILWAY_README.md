# CaperClub Backend - Railway Configuration

## Database Configuration

**ALWAYS USE RAILWAY DATABASE**

This backend is configured to **always use Railway MySQL database**. Local MySQL development is not supported.

### Environment Variables

Set the following environment variable in your Railway deployment:

```bash
CAPERCLUB_DATABASE_URL=mysql://your_railway_connection_string_here
CAPERCLUB_DOOR_LOCK_API_KEY=replace-with-a-long-random-secret
DOOR_LOCK_DELAY_SECONDS=5
```

### Local Development

For local development, you must still use the Railway database. Local MySQL is not supported.

1. Copy `.env.example` to `.env`
2. Set `CAPERCLUB_DATABASE_URL` to your Railway database connection string
3. Run the backend: `uvicorn backend.main:app --reload --port 8001`

### Deployment

The backend is designed to run on Railway with the following configuration:

- **Database**: Railway MySQL (always)
- **Environment**: Production
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`

## Smart Door Lock

The smart lock flow is:

1. Browser face recognition posts authenticated detection results to `/door/detection`
2. Backend writes the latest `LOCK` or `UNLOCK` command into MySQL
3. ESP32 polls `GET /door/status` with the `X-Door-Key` header
4. Relay changes the NC lock state based on the command

Endpoints:

- `GET /door/status` requires `X-Door-Key`
- `POST /door/unlock` requires `X-Door-Key`
- `POST /door/lock` requires `X-Door-Key`
- `GET /door/state` is for authenticated admin UI status only

Do not expose `CAPERCLUB_DOOR_LOCK_API_KEY` in frontend JavaScript.

### Important Notes

- Never use local MySQL for development
- Always use Railway database connection string
- The `.env` file contains the production Railway configuration
- Local database fallbacks are deprecated and should not be used
