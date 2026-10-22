/**
 * frontend/js/app.js
 * ==================
 * Vanilla JavaScript application logic for Industrial AI Inspector.
 * Interacts with the backend REST API via Fetch API.
 */

document.addEventListener("DOMContentLoaded", () => {
    const statusDot = document.getElementById("status-dot");
    const statusLabel = document.getElementById("status-label");
    const statStatus = document.getElementById("stat-status");
    const statPython = document.getElementById("stat-python");
    const statPlatform = document.getElementById("stat-platform");
    const statPhase = document.getElementById("stat-phase");
    const consoleOutput = document.getElementById("console-output");
    const responseTimestamp = document.getElementById("response-timestamp");
    const btnCheckHealth = document.getElementById("btn-check-health");

    /**
     * Poll the backend /api/health endpoint and update UI state.
     */
    async function checkHealth() {
        // Set intermediate loading state
        statusDot.className = "status-dot status-unknown";
        statusLabel.textContent = "Polling Backend...";
        btnCheckHealth.disabled = true;

        try {
            const startTime = performance.now();
            const response = await fetch("/api/health", {
                method: "GET",
                headers: {
                    "Accept": "application/json",
                },
            });

            const latency = Math.round(performance.now() - startTime);

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            const now = new Date();

            // Update status badge
            statusDot.className = "status-dot status-online";
            statusLabel.textContent = `Online (${latency}ms)`;

            // Update stats grid
            statStatus.textContent = data.status.toUpperCase();
            statStatus.style.color = "var(--color-success)";

            if (data.environment) {
                statPython.textContent = `Python ${data.environment.python || "N/A"}`;
                statPlatform.textContent = data.environment.platform || "N/A";
                statPhase.textContent = data.environment.phase ? "Phase 1: Ready" : "Phase 1";
            }

            // Update diagnostic console
            responseTimestamp.textContent = now.toLocaleTimeString();
            consoleOutput.textContent = JSON.stringify(data, null, 2);
            consoleOutput.style.color = "#38bdf8";

        } catch (error) {
            console.error("Health check failure:", error);
            const now = new Date();

            // Update status badge to offline
            statusDot.className = "status-dot status-offline";
            statusLabel.textContent = "Offline / Disconnected";

            // Update stats
            statStatus.textContent = "DISCONNECTED";
            statStatus.style.color = "var(--color-danger)";
            statPython.textContent = "Unknown";
            statPlatform.textContent = "Unknown";

            // Update console
            responseTimestamp.textContent = now.toLocaleTimeString();
            consoleOutput.textContent = `[ERROR] Failed to reach /api/health:\n${error.message}\n\nEnsure Flask backend is running on http://127.0.0.1:5000`;
            consoleOutput.style.color = "#ef4444";
        } finally {
            btnCheckHealth.disabled = false;
        }
    }

    // Attach click handler
    if (btnCheckHealth) {
        btnCheckHealth.addEventListener("click", checkHealth);
    }

    // Run initial pre-flight health check on page load
    checkHealth();
});
