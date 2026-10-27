import http.server
import socketserver
import os

PORT = 8000
LOG_FILE = "diagnostics.log"

class DiagnosticHandler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/api/log":
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            try:
                # Overwrite diagnostics.log so it never grows exponentially
                with open(LOG_FILE, "w") as f:
                    f.write(post_data.decode('utf-8'))
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"Diagnostics saved successfully.")
            except Exception as e:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(str(e).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

with socketserver.TCPServer(("", PORT), DiagnosticHandler) as httpd:
    print(f"[VoxelServer] Running with Diagnostic Logger on port {PORT}...")
    httpd.serve_forever()
