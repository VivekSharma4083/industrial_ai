/**
 * frontend/js/app.js
 * ==================
 * Vanilla JavaScript application controller for Industrial AI Inspector.
 * Coordinates real-time anomaly inspection, dual-model comparison,
 * client-side 60 FPS heatmap opacity blending, history caching, and diagnostics.
 */

document.addEventListener("DOMContentLoaded", () => {
    // --- Application State ---
    const state = {
        selectedFile: null,
        selectedPreset: null,
        selectedModel: "both",          // "reconstruction" | "patchcore" | "both"
        selectedThresholdMode: "unbiased", // "unbiased" | "optimal" | "custom"
        currentInspection: null,
        overlayOpacity: 0.45,
        thresholds: {
            reconstruction: { unbiased: 23.7207, optimal: 23.6698 },
            patchcore: { unbiased: 10.0016, optimal: 13.1616 },
        },
    };

    // --- DOM Elements ---
    // Header & Tabs
    const statusDot = document.getElementById("status-dot");
    const statusLabel = document.getElementById("status-label");
    const navTabs = document.querySelectorAll(".nav-tab");
    const tabPanes = document.querySelectorAll(".tab-pane");
    const historyBadgeCount = document.getElementById("history-badge-count");

    // Input Controls
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("file-input");
    const btnBrowse = document.getElementById("btn-browse");
    const dropzonePrompt = document.getElementById("dropzone-prompt");
    const dropzonePreview = document.getElementById("dropzone-preview");
    const previewThumb = document.getElementById("preview-thumb");
    const previewFilename = document.getElementById("preview-filename");
    const previewFilesize = document.getElementById("preview-filesize");
    const btnClearImage = document.getElementById("btn-clear-image");
    const presetBtns = document.querySelectorAll(".preset-btn");

    // Model & Threshold Selectors
    const modelSelector = document.getElementById("model-selector");
    const modelCaption = document.getElementById("model-caption");
    const thresholdSelector = document.getElementById("threshold-selector");
    const threshReconDisplay = document.getElementById("thresh-recon-display");
    const threshPatchDisplay = document.getElementById("thresh-patch-display");
    const customThresholdsContainer = document.getElementById("custom-thresholds-container");
    const inputThreshRecon = document.getElementById("input-thresh-recon");
    const inputThreshPatch = document.getElementById("input-thresh-patch");
    const sliderReconVal = document.getElementById("slider-recon-val");
    const sliderPatchVal = document.getElementById("slider-patch-val");
    const btnRunInspect = document.getElementById("btn-run-inspect");
    const btnInspectText = document.getElementById("btn-inspect-text");

    // Results Views
    const emptyStateView = document.getElementById("empty-state-view");
    const loadingStateView = document.getElementById("loading-state-view");
    const resultsView = document.getElementById("results-view");
    const verdictBanner = document.getElementById("verdict-banner");
    const verdictIcon = document.getElementById("verdict-icon");
    const verdictStatus = document.getElementById("verdict-status");
    const verdictSubtitle = document.getElementById("verdict-subtitle");
    const verdictLatency = document.getElementById("verdict-latency");
    const verdictAgreement = document.getElementById("verdict-agreement");
    const opacitySlider = document.getElementById("opacity-slider");
    const opacityValDisplay = document.getElementById("opacity-val-display");
    const visualPanelsContainer = document.getElementById("visual-panels-container");

    // History Tab Elements
    const historyContainer = document.getElementById("history-container");
    const historyTable = document.getElementById("history-table");
    const historyTableBody = document.getElementById("history-table-body");
    const historyEmptyMessage = document.getElementById("history-empty-message");
    const btnClearHistory = document.getElementById("btn-clear-history");

    // Diagnostics Tab Elements
    const btnCheckHealth = document.getElementById("btn-check-health");
    const statStatus = document.getElementById("stat-status");
    const statPython = document.getElementById("stat-python");
    const statPlatform = document.getElementById("stat-platform");
    const statPhase = document.getElementById("stat-phase");
    const consoleOutput = document.getElementById("console-output");
    const responseTimestamp = document.getElementById("response-timestamp");

    // =========================================================================
    // 1. Tab Switching Controller
    // =========================================================================
    navTabs.forEach((tab) => {
        tab.addEventListener("click", () => {
            const targetId = tab.getAttribute("data-tab");

            navTabs.forEach((t) => t.classList.remove("active"));
            tabPanes.forEach((p) => p.classList.remove("active"));

            tab.classList.add("active");
            const targetPane = document.getElementById(targetId);
            if (targetPane) {
                targetPane.classList.add("active");
            }

            if (targetId === "tab-history") {
                fetchHistory();
            } else if (targetId === "tab-comparison") {
                fetchEvaluationSummary();
            } else if (targetId === "tab-diagnostics") {
                checkBackendStatus();
            }
        });
    });

    // =========================================================================
    // 2. Dropzone & File Input Handling
    // =========================================================================
    if (btnBrowse) {
        btnBrowse.addEventListener("click", (e) => {
            e.stopPropagation();
            fileInput.click();
        });
    }

    if (dropzone) {
        dropzone.addEventListener("click", () => {
            if (!state.selectedFile && !state.selectedPreset) {
                fileInput.click();
            }
        });

        dropzone.addEventListener("dragover", (e) => {
            e.preventDefault();
            dropzone.classList.add("dragover");
        });

        dropzone.addEventListener("dragleave", () => {
            dropzone.classList.remove("dragover");
        });

        dropzone.addEventListener("drop", (e) => {
            e.preventDefault();
            dropzone.classList.remove("dragover");
            if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                handleSelectedFile(e.dataTransfer.files[0]);
            }
        });
    }

    if (fileInput) {
        fileInput.addEventListener("change", (e) => {
            if (e.target.files && e.target.files.length > 0) {
                handleSelectedFile(e.target.files[0]);
            }
        });
    }

    if (btnClearImage) {
        btnClearImage.addEventListener("click", (e) => {
            e.stopPropagation();
            clearSelectedImage();
        });
    }

    function handleSelectedFile(file) {
        // Validate MIME type
        const validTypes = ["image/png", "image/jpeg", "image/bmp"];
        if (!validTypes.includes(file.type) && !file.name.match(/\.(png|jpe?g|bmp)$/i)) {
            alert("Please select a valid image file (PNG, JPEG, or BMP).");
            return;
        }

        // Validate max size 10 MB
        if (file.size > 10 * 1024 * 1024) {
            alert("Selected file exceeds the maximum 10 MB upload limit.");
            return;
        }

        state.selectedFile = file;
        state.selectedPreset = null;
        unselectPresets();

        // Display thumbnail preview
        const reader = new FileReader();
        reader.onload = (event) => {
            previewThumb.src = event.target.result;
            previewFilename.textContent = file.name;
            previewFilesize.textContent = `${Math.round(file.size / 1024)} KB`;

            dropzonePrompt.classList.add("hidden");
            dropzonePreview.classList.remove("hidden");
            btnRunInspect.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    function clearSelectedImage() {
        state.selectedFile = null;
        state.selectedPreset = null;
        fileInput.value = "";
        previewThumb.src = "";
        dropzonePreview.classList.add("hidden");
        dropzonePrompt.classList.remove("hidden");
        unselectPresets();
        btnRunInspect.disabled = true;
    }

    function unselectPresets() {
        presetBtns.forEach((b) => b.classList.remove("selected"));
    }

    // =========================================================================
    // 3. Preset Test Samples Handling
    // =========================================================================
    presetBtns.forEach((btn) => {
        btn.addEventListener("click", () => {
            const presetId = btn.getAttribute("data-preset");
            unselectPresets();
            btn.classList.add("selected");

            state.selectedPreset = presetId;
            state.selectedFile = null;
            fileInput.value = "";

            // Show preset preview
            const presetName = btn.querySelector(".preset-name").textContent;
            previewFilename.textContent = `Preset: ${presetName}`;
            previewFilesize.textContent = "Server Ground Truth";
            previewThumb.src = `/api/samples/${presetId}`;

            dropzonePrompt.classList.add("hidden");
            dropzonePreview.classList.remove("hidden");
            btnRunInspect.disabled = false;
        });
    });

    // =========================================================================
    // 4. Model & Threshold Selector Logic
    // =========================================================================
    if (modelSelector) {
        modelSelector.querySelectorAll(".pill-btn").forEach((pill) => {
            pill.addEventListener("click", () => {
                modelSelector.querySelectorAll(".pill-btn").forEach((p) => p.classList.remove("active"));
                pill.classList.add("active");
                state.selectedModel = pill.getAttribute("data-value");

                // Update caption
                if (state.selectedModel === "both") {
                    modelCaption.textContent = "Runs both models concurrently to evaluate agreement and spatial contrast.";
                } else if (state.selectedModel === "reconstruction") {
                    modelCaption.textContent = "Spatial convolutional autoencoder trained on ResNet50 layer2 representations.";
                } else if (state.selectedModel === "patchcore") {
                    modelCaption.textContent = "Coreset memory bank with neighborhood patch aggregation from layers 2 and 3.";
                }
            });
        });
    }

    if (thresholdSelector) {
        thresholdSelector.querySelectorAll(".pill-btn").forEach((pill) => {
            pill.addEventListener("click", () => {
                thresholdSelector.querySelectorAll(".pill-btn").forEach((p) => p.classList.remove("active"));
                pill.classList.add("active");
                state.selectedThresholdMode = pill.getAttribute("data-value");

                updateThresholdDisplay();
            });
        });
    }

    function updateThresholdDisplay() {
        const mode = state.selectedThresholdMode;

        if (mode === "custom") {
            customThresholdsContainer.classList.remove("hidden");
            threshReconDisplay.textContent = parseFloat(inputThreshRecon.value).toFixed(2);
            threshPatchDisplay.textContent = parseFloat(inputThreshPatch.value).toFixed(2);
        } else {
            customThresholdsContainer.classList.add("hidden");
            const rVal = state.thresholds.reconstruction[mode] || 23.7207;
            const pVal = state.thresholds.patchcore[mode] || 10.0016;
            threshReconDisplay.textContent = rVal.toFixed(4);
            threshPatchDisplay.textContent = pVal.toFixed(4);
        }
    }

    if (inputThreshRecon) {
        inputThreshRecon.addEventListener("input", (e) => {
            sliderReconVal.textContent = parseFloat(e.target.value).toFixed(2);
            threshReconDisplay.textContent = parseFloat(e.target.value).toFixed(2);
        });
    }

    if (inputThreshPatch) {
        inputThreshPatch.addEventListener("input", (e) => {
            sliderPatchVal.textContent = parseFloat(e.target.value).toFixed(2);
            threshPatchDisplay.textContent = parseFloat(e.target.value).toFixed(2);
        });
    }

    // =========================================================================
    // 5. Interactive Heatmap Opacity Slider (60 FPS Client-Side Blending)
    // =========================================================================
    if (opacitySlider) {
        opacitySlider.addEventListener("input", (e) => {
            const val = parseInt(e.target.value, 10);
            state.overlayOpacity = val / 100.0;
            opacityValDisplay.textContent = `${val}%`;

            // Instantly apply opacity to all active heatmap overlay layers
            const heatmapLayers = document.querySelectorAll(".layer-heatmap");
            heatmapLayers.forEach((layer) => {
                layer.style.opacity = state.overlayOpacity.toString();
            });
        });
    }

    // =========================================================================
    // 6. Visual Inspection Execution
    // =========================================================================
    if (btnRunInspect) {
        btnRunInspect.addEventListener("click", executeInspection);
    }

    async function executeInspection() {
        if (!state.selectedFile && !state.selectedPreset) {
            alert("Please select a product image or preset sample first.");
            return;
        }

        // Set UI loading state
        btnRunInspect.disabled = true;
        btnInspectText.textContent = "Inspecting...";
        emptyStateView.classList.add("hidden");
        resultsView.classList.add("hidden");
        loadingStateView.classList.remove("hidden");

        const formData = new FormData();
        formData.append("model", state.selectedModel);
        formData.append("threshold_mode", state.selectedThresholdMode);

        if (state.selectedThresholdMode === "custom") {
            formData.append("custom_threshold_recon", inputThreshRecon.value);
            formData.append("custom_threshold_patch", inputThreshPatch.value);
        }

        if (state.selectedFile) {
            formData.append("image", state.selectedFile);
        } else if (state.selectedPreset) {
            formData.append("sample_id", state.selectedPreset);
        }

        try {
            const response = await fetch("/api/inspect", {
                method: "POST",
                body: formData,
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.error || `HTTP ${response.status}`);
            }

            const data = await response.json();
            state.currentInspection = data;

            // Render inspection results
            renderInspectionResults(data);
            updateHistoryBadge();

        } catch (error) {
            console.error("Inspection failure:", error);
            alert(`Inspection failed: ${error.message}`);
            emptyStateView.classList.remove("hidden");
        } finally {
            loadingStateView.classList.add("hidden");
            btnRunInspect.disabled = false;
            btnInspectText.textContent = "Run Visual Inspection";
        }
    }

    // =========================================================================
    // 7. Results View Rendering
    // =========================================================================
    function renderInspectionResults(data) {
        resultsView.classList.remove("hidden");

        const summary = data.summary || {};
        const isAnomaly = summary.overall_status
            ? summary.overall_status.includes("DEFECT")
            : (data.reconstruction?.is_anomaly || data.patchcore?.is_anomaly);

        const isDiscrepancy = summary.agreement === "DISCORDANT";

        // Update Verdict Banner
        verdictBanner.className = "verdict-banner";
        if (isDiscrepancy) {
            verdictBanner.classList.add("verdict-discrepancy");
            verdictStatus.textContent = "UNIT SUSPICIOUS / DISCREPANCY";
            verdictSubtitle.textContent = summary.agreement_description || "Models reached conflicting decisions. Manual QA recommended.";
            verdictIcon.innerHTML = `
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <circle cx="12" cy="12" r="10"></circle>
                    <line x1="12" y1="8" x2="12" y2="12"></line>
                    <line x1="12" y1="16" x2="12.01" y2="16"></line>
                </svg>`;
        } else if (isAnomaly) {
            verdictBanner.classList.add("verdict-defect");
            verdictStatus.textContent = "ANOMALOUS / DEFECT DETECTED";
            verdictSubtitle.textContent = summary.agreement_description || "Visual patterns deviate significantly from normal production baseline.";
            verdictIcon.innerHTML = `
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"></path>
                    <line x1="12" y1="9" x2="12" y2="13"></line>
                    <line x1="12" y1="17" x2="12.01" y2="17"></line>
                </svg>`;
        } else {
            verdictBanner.classList.add("verdict-pass");
            verdictStatus.textContent = "NORMAL PRODUCT / PASS";
            verdictSubtitle.textContent = summary.agreement_description || "Surface and structural features conform to normal defect-free manifold.";
            verdictIcon.innerHTML = `
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path>
                    <polyline points="9 12 11 14 15 10"></polyline>
                </svg>`;
        }

        verdictLatency.textContent = summary.total_latency_ms
            ? `${summary.total_latency_ms} ms`
            : `${(data.reconstruction || data.patchcore).latency_ms} ms`;

        verdictAgreement.textContent = summary.agreement
            ? (summary.agreement === "DISCORDANT" ? "Discrepancy" : "Concordant")
            : "Single Model";

        // Render Visual Panels
        visualPanelsContainer.innerHTML = "";

        if (data.model_choice === "both" && data.reconstruction && data.patchcore) {
            renderDualModelComparison(data);
        } else {
            const single = data.reconstruction || data.patchcore;
            renderSingleModelPanels(single, data.preview_image);
        }

        // Apply initial opacity slider setting
        const heatmapLayers = document.querySelectorAll(".layer-heatmap");
        heatmapLayers.forEach((layer) => {
            layer.style.opacity = state.overlayOpacity.toString();
        });
    }

    function renderSingleModelPanels(modelData, previewImgB64) {
        const wrapper = document.createElement("div");
        wrapper.className = "single-model-panels";

        wrapper.innerHTML = `
            <!-- Panel 1: Original Input -->
            <div class="panel-card">
                <div class="panel-header">
                    <span>Original Product Image</span>
                    <span class="panel-tag">Input 224x224</span>
                </div>
                <div class="panel-canvas-wrapper">
                    <img src="${previewImgB64}" alt="Original Input" class="panel-img">
                </div>
            </div>

            <!-- Panel 2: Continuous Heatmap -->
            <div class="panel-card">
                <div class="panel-header">
                    <span>Anomaly Heatmap</span>
                    <span class="panel-tag">Jet Colormap</span>
                </div>
                <div class="panel-canvas-wrapper">
                    <img src="${modelData.heatmap_image}" alt="Anomaly Heatmap" class="panel-img">
                </div>
            </div>

            <!-- Panel 3: Live Interactive Heatmap Overlay -->
            <div class="panel-card">
                <div class="panel-header">
                    <span>Interactive Explainable Overlay</span>
                    <span class="panel-tag">Live Alpha Blend</span>
                </div>
                <div class="panel-canvas-wrapper layered-overlay-wrapper">
                    <img src="${previewImgB64}" alt="Base Product" class="layer-base">
                    <img src="${modelData.heatmap_image}" alt="Heatmap Overlay Layer" class="layer-heatmap">
                </div>
            </div>
        `;

        visualPanelsContainer.appendChild(wrapper);
    }

    function renderDualModelComparison(data) {
        const recon = data.reconstruction;
        const patch = data.patchcore;
        const baseImg = data.preview_image;

        const grid = document.createElement("div");
        grid.className = "dual-model-grid";

        grid.innerHTML = `
            <!-- Left Column: Feature Reconstruction Model -->
            <div class="model-column-card">
                <div class="model-column-header">
                    <div>
                        <div class="model-name-title">Feature Reconstruction Autoencoder</div>
                        <span class="badge badge-subtle">ResNet-50 layer2 &bull; 512-dim</span>
                    </div>
                    <span class="status-pill ${recon.is_anomaly ? 'pill-defect' : 'pill-normal'}">${recon.decision}</span>
                </div>
                <div class="score-strip">
                    <div class="strip-item">
                        <span class="strip-label">Score</span>
                        <span class="strip-val ${recon.is_anomaly ? 'val-defect' : 'val-pass'}">${recon.score}</span>
                    </div>
                    <div class="strip-item">
                        <span class="strip-label">Threshold &tau;</span>
                        <span class="strip-val">${recon.threshold}</span>
                    </div>
                    <div class="strip-item">
                        <span class="strip-label">Margin</span>
                        <span class="strip-val">${recon.score_margin > 0 ? '+' : ''}${recon.score_margin}</span>
                    </div>
                    <div class="strip-item">
                        <span class="strip-label">Latency</span>
                        <span class="strip-val">${recon.latency_ms} ms</span>
                    </div>
                </div>
                <div class="model-visual-row">
                    <div class="panel-card">
                        <div class="panel-header">
                            <span>Reconstruction Heatmap</span>
                        </div>
                        <div class="panel-canvas-wrapper">
                            <img src="${recon.heatmap_image}" alt="Reconstruction Heatmap" class="panel-img">
                        </div>
                    </div>
                    <div class="panel-card">
                        <div class="panel-header">
                            <span>Interactive Overlay</span>
                        </div>
                        <div class="panel-canvas-wrapper layered-overlay-wrapper">
                            <img src="${baseImg}" alt="Base Image" class="layer-base">
                            <img src="${recon.heatmap_image}" alt="Overlay Heatmap" class="layer-heatmap">
                        </div>
                    </div>
                </div>
            </div>

            <!-- Right Column: PatchCore Baseline -->
            <div class="model-column-card">
                <div class="model-column-header">
                    <div>
                        <div class="model-name-title">PatchCore Baseline</div>
                        <span class="badge badge-accent">layer2 + layer3 &bull; 1536-dim</span>
                    </div>
                    <span class="status-pill ${patch.is_anomaly ? 'pill-defect' : 'pill-normal'}">${patch.decision}</span>
                </div>
                <div class="score-strip">
                    <div class="strip-item">
                        <span class="strip-label">Score</span>
                        <span class="strip-val ${patch.is_anomaly ? 'val-defect' : 'val-pass'}">${patch.score}</span>
                    </div>
                    <div class="strip-item">
                        <span class="strip-label">Threshold &tau;</span>
                        <span class="strip-val">${patch.threshold}</span>
                    </div>
                    <div class="strip-item">
                        <span class="strip-label">Margin</span>
                        <span class="strip-val">${patch.score_margin > 0 ? '+' : ''}${patch.score_margin}</span>
                    </div>
                    <div class="strip-item">
                        <span class="strip-label">Latency</span>
                        <span class="strip-val">${patch.latency_ms} ms</span>
                    </div>
                </div>
                <div class="model-visual-row">
                    <div class="panel-card">
                        <div class="panel-header">
                            <span>PatchCore Heatmap</span>
                        </div>
                        <div class="panel-canvas-wrapper">
                            <img src="${patch.heatmap_image}" alt="PatchCore Heatmap" class="panel-img">
                        </div>
                    </div>
                    <div class="panel-card">
                        <div class="panel-header">
                            <span>Interactive Overlay</span>
                        </div>
                        <div class="panel-canvas-wrapper layered-overlay-wrapper">
                            <img src="${baseImg}" alt="Base Image" class="layer-base">
                            <img src="${patch.heatmap_image}" alt="Overlay Heatmap" class="layer-heatmap">
                        </div>
                    </div>
                </div>
            </div>
        `;

        visualPanelsContainer.appendChild(grid);

        // Comparative Analysis Card
        const analysisCard = document.createElement("div");
        analysisCard.className = "comparison-analysis-card";

        let analysisText = "";
        if (recon.is_anomaly && patch.is_anomaly) {
            analysisText = "Both models detected anomalous visual patterns with high confidence. PatchCore exhibits tighter spatial localization around defect regions with reduced contour edge noise.";
        } else if (!recon.is_anomaly && !patch.is_anomaly) {
            analysisText = "Both models passed this unit as compliant with the normal manufacturing baseline. Feature maps and patch embeddings fall well within the normal distribution.";
        } else {
            const flagging = recon.is_anomaly ? "Feature Reconstruction" : "PatchCore";
            const passing = recon.is_anomaly ? "PatchCore" : "Feature Reconstruction";
            analysisText = `${flagging} flagged this unit, while ${passing} evaluated it as normal. This discrepancy typically occurs on faint, low-contrast contamination or along sharp glass refraction contours.`;
        }

        analysisCard.innerHTML = `
            <div class="analysis-title">Comparative Localization & Engineering Analysis</div>
            <p class="analysis-text">${analysisText}</p>
        `;

        visualPanelsContainer.appendChild(analysisCard);
    }

    // =========================================================================
    // 8. Inspection History Management
    // =========================================================================
    async function fetchHistory() {
        try {
            const res = await fetch("/api/history?limit=50");
            if (!res.ok) return;

            const data = await res.json();
            const items = data.history || [];

            if (historyBadgeCount) {
                historyBadgeCount.textContent = items.length.toString();
            }

            if (items.length === 0) {
                historyEmptyMessage.classList.remove("hidden");
                historyTable.classList.add("hidden");
                return;
            }

            historyEmptyMessage.classList.add("hidden");
            historyTable.classList.remove("hidden");
            historyTableBody.innerHTML = "";

            items.forEach((item) => {
                const tr = document.createElement("tr");
                const isDefect = item.status && item.status.includes("DEFECT");
                const isDiscrepancy = item.status && item.status.includes("DISCREPANCY");
                let statusClass = "pill-normal";
                if (isDiscrepancy) statusClass = "pill-discrepancy";
                else if (isDefect) statusClass = "pill-defect";

                const dt = new Date(item.timestamp);
                const timeStr = dt.toLocaleTimeString();

                tr.innerHTML = `
                    <td><img src="${item.thumbnail || ''}" alt="Thumb" class="history-thumb"></td>
                    <td class="metric-num">${timeStr}</td>
                    <td>${item.source_name}</td>
                    <td><span class="badge badge-subtle">${item.model_choice}</span></td>
                    <td class="metric-num">${item.recon_score !== null ? item.recon_score.toFixed(2) : '--'}</td>
                    <td class="metric-num">${item.patch_score !== null ? item.patch_score.toFixed(2) : '--'}</td>
                    <td><span class="status-pill ${statusClass}">${item.status}</span></td>
                    <td><button type="button" class="btn btn-sm btn-secondary btn-reopen" data-id="${item.id}">View</button></td>
                `;

                const btnView = tr.querySelector(".btn-reopen");
                btnView.addEventListener("click", () => reopenHistoryItem(item.id));

                historyTableBody.appendChild(tr);
            });
        } catch (err) {
            console.error("Failed to fetch history:", err);
        }
    }

    async function reopenHistoryItem(id) {
        try {
            const res = await fetch(`/api/history/${id}`);
            if (!res.ok) throw new Error("History item not found");
            const data = await res.json();

            // Switch to inspection tab
            const inspectTab = document.getElementById("btn-tab-inspect");
            if (inspectTab) inspectTab.click();

            // Render inspection data
            emptyStateView.classList.add("hidden");
            renderInspectionResults(data);
        } catch (err) {
            alert(`Failed to load history item: ${err.message}`);
        }
    }

    async function updateHistoryBadge() {
        try {
            const res = await fetch("/api/history?limit=1");
            if (!res.ok) return;
            const data = await res.json();
            if (historyBadgeCount) {
                historyBadgeCount.textContent = data.count.toString();
            }
        } catch (e) {
            // Silently ignore badge fetch errors
        }
    }

    if (btnClearHistory) {
        btnClearHistory.addEventListener("click", async () => {
            if (!confirm("Are you sure you want to clear all inspection history?")) return;
            try {
                await fetch("/api/history", { method: "DELETE" });
                fetchHistory();
                if (historyBadgeCount) historyBadgeCount.textContent = "0";
            } catch (err) {
                alert(`Failed to clear history: ${err.message}`);
            }
        });
    }

    // =========================================================================
    // 9. Model Benchmarks Fetcher
    // =========================================================================
    async function fetchEvaluationSummary() {
        try {
            const res = await fetch("/api/eval-summary");
            if (!res.ok) return;
            const data = await res.json();

            const comp = data.comparison || {};
            const rAuroc = comp.image_roc_auc?.reconstruction || 0.9968;
            const pAuroc = comp.image_roc_auc?.patchcore || 1.0;
            const rPix = comp.pixel_roc_auc?.reconstruction || 0.9641;
            const pPix = comp.pixel_roc_auc?.patchcore || 0.9797;
            const rLat = comp.mean_cpu_latency_ms?.reconstruction || 160.18;
            const pLat = comp.mean_cpu_latency_ms?.patchcore || 224.46;
            const rMiss = comp.false_negatives?.reconstruction ?? 3;
            const pMiss = comp.false_negatives?.patchcore ?? 0;

            const benchImgAuroc = document.getElementById("bench-image-auroc");
            const benchPixAuroc = document.getElementById("bench-pixel-auroc");
            const benchLatency = document.getElementById("bench-latency");
            const benchMisses = document.getElementById("bench-misses");

            if (benchImgAuroc) benchImgAuroc.textContent = `${(rAuroc * 100).toFixed(2)}% vs ${(pAuroc * 100).toFixed(2)}%`;
            if (benchPixAuroc) benchPixAuroc.textContent = `${(rPix * 100).toFixed(2)}% vs ${(pPix * 100).toFixed(2)}%`;
            if (benchLatency) benchLatency.textContent = `${rLat.toFixed(1)} ms vs ${pLat.toFixed(1)} ms`;
            if (benchMisses) benchMisses.textContent = `${rMiss} vs ${pMiss}`;
        } catch (err) {
            console.error("Failed to load evaluation summary:", err);
        }
    }

    // =========================================================================
    // 10. System Diagnostics & Health Check
    // =========================================================================
    async function checkBackendStatus() {
        statusDot.className = "status-dot status-unknown";
        statusLabel.textContent = "Polling Backend...";
        if (btnCheckHealth) btnCheckHealth.disabled = true;

        try {
            const t0 = performance.now();
            const [healthRes, modelsRes] = await Promise.all([
                fetch("/api/health"),
                fetch("/api/models"),
            ]);

            const latency = Math.round(performance.now() - t0);

            if (!healthRes.ok || !modelsRes.ok) {
                throw new Error("Backend response error");
            }

            const healthData = await healthRes.json();
            const modelsData = await modelsRes.json();

            // Update header status dot
            statusDot.className = "status-dot status-online";
            statusLabel.textContent = `Online (${latency}ms)`;

            // Update stats grid in diagnostics tab
            if (statStatus) {
                statStatus.textContent = healthData.status.toUpperCase();
                statStatus.style.color = "var(--color-success)";
            }
            if (statPython && healthData.environment) {
                statPython.textContent = `Python ${healthData.environment.python || "N/A"}`;
            }
            if (statPlatform && healthData.environment) {
                statPlatform.textContent = healthData.environment.platform || "N/A";
            }
            if (statPhase) {
                statPhase.textContent = "Phase 6: Full-Stack QA";
            }

            // Update console log
            if (responseTimestamp) responseTimestamp.textContent = new Date().toLocaleTimeString();
            if (consoleOutput) {
                consoleOutput.textContent = JSON.stringify({ health: healthData, models: modelsData }, null, 2);
                consoleOutput.style.color = "var(--color-info)";
            }

            // Sync threshold state from models API
            if (modelsData.models) {
                if (modelsData.models.reconstruction?.thresholds) {
                    state.thresholds.reconstruction = modelsData.models.reconstruction.thresholds;
                }
                if (modelsData.models.patchcore?.thresholds) {
                    state.thresholds.patchcore = modelsData.models.patchcore.thresholds;
                }
                updateThresholdDisplay();
            }

        } catch (err) {
            console.error("Health check error:", err);
            statusDot.className = "status-dot status-offline";
            statusLabel.textContent = "Offline / Disconnected";

            if (statStatus) {
                statStatus.textContent = "DISCONNECTED";
                statStatus.style.color = "var(--color-danger)";
            }
            if (responseTimestamp) responseTimestamp.textContent = new Date().toLocaleTimeString();
            if (consoleOutput) {
                consoleOutput.textContent = `[ERROR] Failed to connect to Flask backend:\n${err.message}\n\nEnsure Flask server is running at http://127.0.0.1:5000`;
                consoleOutput.style.color = "var(--color-danger)";
            }
        } finally {
            if (btnCheckHealth) btnCheckHealth.disabled = false;
        }
    }

    // Alias for test compatibility
    const checkHealth = checkBackendStatus;

    if (btnCheckHealth) {
        btnCheckHealth.addEventListener("click", checkHealth);
    }

    // --- Initial Application Bootstrapping ---
    checkHealth();
    updateHistoryBadge();
    fetchEvaluationSummary();
});
