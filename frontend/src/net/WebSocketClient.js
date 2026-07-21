/**
 * WebSocket client — receives landmark + effect-state stream from the server.
 *
 * Handles connection lifecycle, automatic reconnection, and message parsing.
 */

export class WebSocketClient {
    constructor(url, reconnectIntervalMs = 3000) {
        this.url = url;
        this.reconnectIntervalMs = reconnectIntervalMs;
        this._ws = null;
        this._messageHandlers = [];
        this._openHandlers = [];
        this._closeHandlers = [];
        this._shouldReconnect = true;
    }

    connect() {
        this._ws = new WebSocket(this.url);

        this._ws.onopen = () => {
            console.log('[WS] Connected to', this.url);
            this._openHandlers.forEach((h) => h());
        };

        this._ws.onmessage = (event) => {
            try {
                const msg = JSON.parse(event.data);
                this._messageHandlers.forEach((h) => h(msg));
            } catch (err) {
                console.warn('[WS] Failed to parse message:', err);
            }
        };

        this._ws.onclose = () => {
            console.log('[WS] Disconnected');
            this._closeHandlers.forEach((h) => h());
            if (this._shouldReconnect) {
                setTimeout(() => this.connect(), this.reconnectIntervalMs);
            }
        };

        this._ws.onerror = (err) => {
            console.error('[WS] Error:', err);
            this._ws.close();
        };
    }

    disconnect() {
        this._shouldReconnect = false;
        if (this._ws) {
            this._ws.close();
        }
    }

    send(data) {
        if (this._ws && this._ws.readyState === WebSocket.OPEN) {
            this._ws.send(JSON.stringify(data));
        }
    }

    onMessage(handler) { this._messageHandlers.push(handler); }
    onOpen(handler) { this._openHandlers.push(handler); }
    onClose(handler) { this._closeHandlers.push(handler); }
}
