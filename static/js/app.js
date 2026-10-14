/**
 * OmniVision YOLOv8 Prototype - Frontend Application Logic
 * Interactive Canvas, Live Webcam Stream & Real-time Spatial Perception
 */

(() => {
  'use strict';

  // Application State
  const state = {
    mode: 'image', // 'image' | 'webcam'
    currentImageElement: null,
    currentRawFile: null,
    modelInfo: null,
    detections: [],
    metrics: null,
    hoveredDetectionId: null,
    selectedClasses: new Set(), // Set of class IDs to filter, or empty for all
    
    // Config
    confidence: 0.25,
    iou: 0.45,
    renderSettings: {
      showLabels: true,
      showConf: true,
      showBoxes: true,
      showFill: true
    },

    // Webcam
    webcamStream: null,
    webcamActive: false,
    webcamIntervalId: null,
    streamIntervalMs: 200,
    ws: null,
    webcamFps: 0,
    lastFrameTime: performance.now(),
    frameCount: 0
  };

  // DOM Elements Cache
  const el = {
    // App Container
    appContainer: document.querySelector('.app-container'),

    // Tabs & Modes
    tabImage: document.getElementById('tab-image-mode'),
    tabWebcam: document.getElementById('tab-webcam-mode'),
    imageStage: document.getElementById('image-stage-view'),
    webcamStage: document.getElementById('webcam-stage-view'),
    samplePresetsGroup: document.getElementById('sample-presets-group'),
    btnToggleExpand: document.getElementById('btn-toggle-expand'),
    expandBtnText: document.getElementById('expand-btn-text'),

    // Presets
    presetTraffic: document.getElementById('preset-traffic'),
    presetLivingroom: document.getElementById('preset-livingroom'),

    // Dropzone & Canvas
    dropZone: document.getElementById('drop-zone'),
    fileInput: document.getElementById('image-file-input'),
    canvasViewport: document.getElementById('canvas-viewport'),
    canvasContainer: document.getElementById('canvas-container'),
    displayCanvas: document.getElementById('display-canvas'),
    boxTooltip: document.getElementById('box-tooltip'),
    resolutionHud: document.getElementById('canvas-resolution-hud'),
    detectionsHud: document.getElementById('canvas-detections-hud'),

    // Webcam elements
    webcamVideo: document.getElementById('webcam-video'),
    webcamCanvas: document.getElementById('webcam-canvas'),
    btnStartWebcam: document.getElementById('btn-start-webcam'),
    btnStopWebcam: document.getElementById('btn-stop-webcam'),
    webcamIdleOverlay: document.getElementById('webcam-idle-overlay'),
    webcamFpsHud: document.getElementById('webcam-fps-hud'),
    webcamLatencyHud: document.getElementById('webcam-latency-hud'),
    streamIntervalSlider: document.getElementById('stream-interval-slider'),
    streamIntervalVal: document.getElementById('stream-interval-val'),

    // Spinner
    loadingOverlay: document.getElementById('loading-overlay'),
    loadingText: document.getElementById('loading-status-text'),

    // Footer
    btnReupload: document.getElementById('btn-reupload'),
    btnResetFilters: document.getElementById('btn-reset-filters'),
    btnExportJson: document.getElementById('btn-export-json'),
    btnDownloadAnnotated: document.getElementById('btn-download-annotated'),

    // Telemetry
    valInferenceMs: document.getElementById('val-inference-ms'),
    valPrepostMs: document.getElementById('val-prepost-ms'),
    valTotalObjects: document.getElementById('val-total-objects'),
    valFps: document.getElementById('val-fps'),
    quickLatencyText: document.getElementById('quick-latency-text'),

    // Controls
    sliderConf: document.getElementById('slider-conf'),
    valConfText: document.getElementById('val-conf-text'),
    sliderIou: document.getElementById('slider-iou'),
    valIouText: document.getElementById('val-iou-text'),

    // Toggles
    toggleLabels: document.getElementById('toggle-labels'),
    toggleConf: document.getElementById('toggle-conf'),
    toggleBoxes: document.getElementById('toggle-boxes'),
    toggleFill: document.getElementById('toggle-fill'),

    // Classes
    classSearchInput: document.getElementById('class-search-input'),
    classChipsContainer: document.getElementById('class-chips-container'),
    btnSelectAllClasses: document.getElementById('btn-select-all-classes'),

    // Results
    resultsCountBadge: document.getElementById('results-count-badge'),
    classSummaryBar: document.getElementById('class-summary-bar'),
    detectionsList: document.getElementById('detections-list'),
    detectionsEmptyState: document.getElementById('detections-empty-state'),

    // Category Matrix
    catCountPerson: document.getElementById('cat-count-person'),
    catCountObject: document.getElementById('cat-count-object'),
    catCountOther: document.getElementById('cat-count-other'),
    catStatusPerson: document.getElementById('cat-status-person'),
    catStatusObject: document.getElementById('cat-status-object'),
    catStatusOther: document.getElementById('cat-status-other'),
    cardPerson: document.getElementById('card-person'),
    cardObject: document.getElementById('card-object'),
    cardOther: document.getElementById('card-other'),

    // Webcam Category HUD
    webcamLiveHud: document.getElementById('webcam-live-hud'),
    webcamHudPerson: document.getElementById('webcam-hud-person'),
    webcamHudObject: document.getElementById('webcam-hud-object'),
    webcamHudOther: document.getElementById('webcam-hud-other')
  };

  // Canvas context
  const ctx = el.displayCanvas.getContext('2d');
  const webcamCtx = el.webcamCanvas.getContext('2d');

  // ==========================================================================
  // Initialization & Model Info Fetch
  // ==========================================================================

  async function init() {
    setupEventListeners();
    await fetchModelInfo();
  }

  async function fetchModelInfo() {
    try {
      const res = await fetch('/api/info');
      if (res.ok) {
        state.modelInfo = await res.json();
        renderClassChips(state.modelInfo.classes);
      }
    } catch (err) {
      console.warn('Could not fetch model info:', err);
    }
  }

  function renderClassChips(classesList) {
    el.classChipsContainer.innerHTML = '';
    classesList.forEach(cls => {
      const chip = document.createElement('span');
      chip.className = 'class-chip active';
      chip.textContent = cls.name;
      chip.dataset.classId = cls.id;
      chip.addEventListener('click', () => toggleClassFilter(cls.id, chip));
      el.classChipsContainer.appendChild(chip);
    });
  }

  function toggleClassFilter(classId, chipElem) {
    if (state.selectedClasses.size === 0) {
      // First click: switch from "all selected" to "only this one selected"
      state.modelInfo.classes.forEach(c => {
        if (c.id !== classId) state.selectedClasses.add(c.id);
      });
      // The clicked one is kept active, others deactivated
      document.querySelectorAll('.class-chip').forEach(c => {
        if (parseInt(c.dataset.classId) !== classId) c.classList.remove('active');
      });
      el.btnSelectAllClasses.textContent = 'Select All';
    } else {
      if (state.selectedClasses.has(classId)) {
        state.selectedClasses.delete(classId);
        chipElem.classList.add('active');
      } else {
        state.selectedClasses.add(classId);
        chipElem.classList.remove('active');
      }
    }

    // Trigger re-detection if an image is loaded
    if (state.currentImageElement && state.currentRawFile) {
      runDetectionOnFile(state.currentRawFile);
    }
  }

  // ==========================================================================
  // UI Event Listeners
  // ==========================================================================

  function setupEventListeners() {
    // Mode Switcher Tabs
    el.tabImage.addEventListener('click', () => switchMode('image'));
    el.tabWebcam.addEventListener('click', () => switchMode('webcam'));

    // Sample Presets
    el.presetTraffic.addEventListener('click', () => loadSamplePreset('/static/samples/street_traffic.jpg'));
    el.presetLivingroom.addEventListener('click', () => loadSamplePreset('/static/samples/living_room.jpg'));

    // File Drag & Drop
    const dropZone = el.dropZone;
    ['dragenter', 'dragover'].forEach(eventName => {
      dropZone.addEventListener(eventName, e => {
        e.preventDefault();
        dropZone.classList.add('drag-active');
      });
    });

    ['dragleave', 'drop'].forEach(eventName => {
      dropZone.addEventListener(eventName, e => {
        e.preventDefault();
        dropZone.classList.remove('drag-active');
      });
    });

    dropZone.addEventListener('drop', e => {
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        handleFileSelection(e.dataTransfer.files[0]);
      }
    });

    el.fileInput.addEventListener('change', e => {
      if (e.target.files && e.target.files[0]) {
        handleFileSelection(e.target.files[0]);
      }
    });

    // Re-upload button
    el.btnReupload.addEventListener('click', () => {
      el.fileInput.click();
    });

    // Sliders
    el.sliderConf.addEventListener('input', e => {
      state.confidence = parseFloat(e.target.value);
      el.valConfText.textContent = `${state.confidence.toFixed(2)} (${Math.round(state.confidence * 100)}%)`;
      debounceRedetection();
    });

    el.sliderIou.addEventListener('input', e => {
      state.iou = parseFloat(e.target.value);
      el.valIouText.textContent = state.iou.toFixed(2);
      debounceRedetection();
    });

    // Render Toggles
    el.toggleLabels.addEventListener('change', e => {
      state.renderSettings.showLabels = e.target.checked;
      renderCanvas();
    });
    el.toggleConf.addEventListener('change', e => {
      state.renderSettings.showConf = e.target.checked;
      renderCanvas();
    });
    el.toggleBoxes.addEventListener('change', e => {
      state.renderSettings.showBoxes = e.target.checked;
      renderCanvas();
    });
    el.toggleFill.addEventListener('change', e => {
      state.renderSettings.showFill = e.target.checked;
      renderCanvas();
    });

    // Class Search Filter
    el.classSearchInput.addEventListener('input', e => {
      const q = e.target.value.toLowerCase().trim();
      document.querySelectorAll('.class-chip').forEach(chip => {
        const text = chip.textContent.toLowerCase();
        chip.style.display = text.includes(q) ? 'inline-block' : 'none';
      });
    });

    // Select All Classes
    el.btnSelectAllClasses.addEventListener('click', () => {
      state.selectedClasses.clear();
      document.querySelectorAll('.class-chip').forEach(c => c.classList.add('active'));
      if (state.currentImageElement && state.currentRawFile) {
        runDetectionOnFile(state.currentRawFile);
      }
    });

    // Reset Controls
    el.btnResetFilters.addEventListener('click', () => {
      state.confidence = 0.25;
      state.iou = 0.45;
      el.sliderConf.value = 0.25;
      el.sliderIou.value = 0.45;
      el.valConfText.textContent = '0.25 (25%)';
      el.valIouText.textContent = '0.45';
      state.selectedClasses.clear();
      document.querySelectorAll('.class-chip').forEach(c => c.classList.add('active'));
      if (state.currentImageElement && state.currentRawFile) {
        runDetectionOnFile(state.currentRawFile);
      }
    });

    // Export Buttons
    el.btnExportJson.addEventListener('click', exportResultsJson);
    el.btnDownloadAnnotated.addEventListener('click', downloadAnnotatedImage);

    // Interactive Canvas Mouse Hover for Boxes
    el.displayCanvas.addEventListener('mousemove', handleCanvasMouseMove);
    el.displayCanvas.addEventListener('mouseleave', () => {
      state.hoveredDetectionId = null;
      el.boxTooltip.classList.add('hidden');
      renderCanvas();
      clearHighlightCards();
    });

    // Webcam Controls
    el.btnStartWebcam.addEventListener('click', startWebcam);
    el.btnStopWebcam.addEventListener('click', stopWebcam);
    el.streamIntervalSlider.addEventListener('input', e => {
      state.streamIntervalMs = parseInt(e.target.value);
      el.streamIntervalVal.textContent = `${state.streamIntervalMs}ms`;
      if (state.webcamActive) {
        restartWebcamInterval();
      }
    });

    // Expand / Theater Mode Screen Toggle
    if (el.btnToggleExpand) {
      el.btnToggleExpand.addEventListener('click', () => {
        const isExpanded = el.appContainer.classList.toggle('theater-mode');
        if (el.expandBtnText) {
          el.expandBtnText.textContent = isExpanded ? 'Standard View' : 'Expand Screen';
        }
        if (state.mode === 'image' && state.currentImageElement) {
          setTimeout(renderCanvas, 200);
        }
      });
    }
  }

  let debounceTimer = null;
  function debounceRedetection() {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
      if (state.mode === 'image' && state.currentRawFile) {
        runDetectionOnFile(state.currentRawFile);
      }
    }, 280);
  }

  // ==========================================================================
  // Mode Switching (Image vs Webcam)
  // ==========================================================================

  function switchMode(newMode) {
    state.mode = newMode;
    if (newMode === 'image') {
      el.tabImage.classList.add('active');
      el.tabWebcam.classList.remove('active');
      el.imageStage.classList.remove('hidden');
      el.webcamStage.classList.add('hidden');
      el.samplePresetsGroup.style.display = 'flex';
      stopWebcam();
    } else {
      el.tabWebcam.classList.add('active');
      el.tabImage.classList.remove('active');
      el.webcamStage.classList.remove('hidden');
      el.imageStage.classList.add('hidden');
      el.samplePresetsGroup.style.display = 'none';
    }
  }

  // ==========================================================================
  // Image Loading & Detection
  // ==========================================================================

  function handleFileSelection(file) {
    if (!file.type.startsWith('image/')) {
      alert('Please select a valid image file (PNG, JPG, etc.).');
      return;
    }

    state.currentRawFile = file;
    const reader = new FileReader();
    reader.onload = e => {
      const img = new Image();
      img.onload = () => {
        state.currentImageElement = img;
        showCanvasView(img);
        runDetectionOnFile(file);
      };
      img.src = e.target.result;
    };
    reader.readAsDataURL(file);
  }

  async function loadSamplePreset(imageUrl) {
    showLoading(true, 'Loading sample preset image...');
    try {
      const res = await fetch(imageUrl);
      const blob = await res.blob();
      const file = new File([blob], 'sample.jpg', { type: blob.type });
      handleFileSelection(file);
    } catch (err) {
      alert('Failed to load sample preset: ' + err.message);
      showLoading(false);
    }
  }

  function showCanvasView(img) {
    el.dropZone.classList.add('hidden');
    el.canvasViewport.classList.remove('hidden');
    el.btnReupload.disabled = false;

    // Set canvas internal resolution to match natural image
    el.displayCanvas.width = img.naturalWidth;
    el.displayCanvas.height = img.naturalHeight;
    el.resolutionHud.textContent = `${img.naturalWidth} × ${img.naturalHeight}px`;

    renderCanvas();
  }

  async function runDetectionOnFile(file) {
    showLoading(true, 'Running YOLOv8 inference...');

    const formData = new FormData();
    formData.append('file', file);
    formData.append('confidence', state.confidence);
    formData.append('iou', state.iou);

    // If some classes are excluded
    if (state.selectedClasses.size > 0 && state.modelInfo) {
      const allowed = state.modelInfo.classes
        .filter(c => !state.selectedClasses.has(c.id))
        .map(c => c.id);
      formData.append('classes', allowed.join(','));
    }

    try {
      const res = await fetch('/api/detect', {
        method: 'POST',
        body: formData
      });

      if (!res.ok) throw new Error(`Server returned ${res.status}`);
      const data = await res.json();
      
      state.detections = data.detections || [];
      state.metrics = data.metrics;
      
      updateTelemetry(data.metrics, state.detections.length);
      updateDetectionsUI(data.detections, data.class_counts, data.category_counts);
      renderCanvas();

      el.btnExportJson.disabled = false;
      el.btnDownloadAnnotated.disabled = false;
    } catch (err) {
      console.error('Detection error:', err);
      alert('Inference error: ' + err.message);
    } finally {
      showLoading(false);
    }
  }

  function updateCategoryMatrix(categoryCounts, detections) {
    const pCount = categoryCounts ? (categoryCounts.person || 0) : 0;
    const oCount = categoryCounts ? (categoryCounts.object || 0) : 0;
    const sCount = categoryCounts ? (categoryCounts.something_else || 0) : 0;

    if (el.catCountPerson) el.catCountPerson.textContent = pCount;
    if (el.catCountObject) el.catCountObject.textContent = oCount;
    if (el.catCountOther) el.catCountOther.textContent = sCount;

    if (el.cardPerson) el.cardPerson.classList.toggle('has-items', pCount > 0);
    if (el.cardObject) el.cardObject.classList.toggle('has-items', oCount > 0);
    if (el.cardOther) el.cardOther.classList.toggle('has-items', sCount > 0);

    if (el.catStatusPerson) el.catStatusPerson.textContent = pCount > 0 ? `${pCount} in frame` : 'None detected';
    if (el.catStatusObject) el.catStatusObject.textContent = oCount > 0 ? `${oCount} in frame` : 'None detected';
    if (el.catStatusOther) el.catStatusOther.textContent = sCount > 0 ? `${sCount} in frame` : 'None detected';

    // Webcam HUD live counters
    if (el.webcamHudPerson) el.webcamHudPerson.textContent = `👤 ${pCount} Person${pCount === 1 ? '' : 's'}`;
    if (el.webcamHudObject) el.webcamHudObject.textContent = `📦 ${oCount} Object${oCount === 1 ? '' : 's'}`;
    if (el.webcamHudOther) el.webcamHudOther.textContent = `⚡ ${sCount} Other`;
  }

  function updateTelemetry(metrics, totalObjects) {
    if (!metrics) return;
    el.valInferenceMs.textContent = metrics.inference_ms;
    el.valPrepostMs.textContent = (metrics.preprocess_ms + metrics.postprocess_ms).toFixed(1);
    el.valTotalObjects.textContent = totalObjects;
    el.valFps.textContent = metrics.fps;
    el.quickLatencyText.textContent = `${metrics.total_ms} ms`;
    el.detectionsHud.textContent = `${totalObjects} Objects`;
  }

  function updateDetectionsUI(detections, classCounts, categoryCounts) {
    updateCategoryMatrix(categoryCounts, detections);
    el.resultsCountBadge.textContent = `${detections.length} items`;

    // Render Class Breakdown Pills
    el.classSummaryBar.innerHTML = '';
    const entries = Object.entries(classCounts || {}).sort((a, b) => b[1] - a[1]);
    if (entries.length === 0) {
      el.classSummaryBar.innerHTML = '<span class="empty-hint">No objects detected</span>';
    } else {
      entries.forEach(([clsName, count]) => {
        const pill = document.createElement('div');
        pill.className = 'summary-pill';
        const detItem = detections.find(d => d.class_name === clsName);
        const color = detItem ? detItem.color : '#00f0ff';
        pill.innerHTML = `
          <span class="summary-dot" style="background:${color}; box-shadow: 0 0 6px ${color}"></span>
          <span>${clsName}</span>
          <span class="summary-count">${count}</span>
        `;
        el.classSummaryBar.appendChild(pill);
      });
    }

    // Render Detection Cards List
    el.detectionsList.innerHTML = '';
    if (detections.length === 0) {
      el.detectionsList.innerHTML = `
        <div class="empty-state">
          <p>No objects found above confidence ${Math.round(state.confidence * 100)}%.</p>
        </div>`;
      return;
    }

    detections.forEach(det => {
      const card = document.createElement('div');
      card.className = 'detection-card';
      card.id = `det-card-${det.id}`;
      card.dataset.id = det.id;

      card.innerHTML = `
        <div class="det-card-left">
          <span class="det-card-color" style="background:${det.color}; box-shadow: 0 0 6px ${det.color}"></span>
          <div>
            <div class="det-card-name">${det.display_name || det.class_name}</div>
            <div class="det-card-box">[${det.box.x1}, ${det.box.y1}, ${det.box.width}×${det.box.height}]</div>
          </div>
        </div>
        <div class="det-card-right">
          <div class="det-card-conf">${(det.confidence * 100).toFixed(1)}%</div>
          <div class="conf-bar-mini">
            <div class="conf-bar-fill" style="width: ${det.confidence * 100}%; background: ${det.color}"></div>
          </div>
        </div>
      `;

      // Hover card highlights canvas box
      card.addEventListener('mouseenter', () => {
        state.hoveredDetectionId = det.id;
        renderCanvas();
      });
      card.addEventListener('mouseleave', () => {
        state.hoveredDetectionId = null;
        renderCanvas();
      });

      el.detectionsList.appendChild(card);
    });
  }

  function highlightCard(id) {
    document.querySelectorAll('.detection-card').forEach(c => c.classList.remove('active-hover'));
    const target = document.getElementById(`det-card-${id}`);
    if (target) {
      target.classList.add('active-hover');
      target.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }

  function clearHighlightCards() {
    document.querySelectorAll('.detection-card').forEach(c => c.classList.remove('active-hover'));
  }

  // ==========================================================================
  // Interactive Canvas Rendering
  // ==========================================================================

  function renderCanvas() {
    if (!state.currentImageElement) return;

    const w = el.displayCanvas.width;
    const h = el.displayCanvas.height;

    ctx.clearRect(0, 0, w, h);
    ctx.drawImage(state.currentImageElement, 0, 0, w, h);

    if (!state.renderSettings.showBoxes || !state.detections.length) return;

    state.detections.forEach(det => {
      const isHovered = state.hoveredDetectionId === det.id;
      const b = det.box;
      const color = det.color;

      // 1. Semi-transparent filled overlay
      if (state.renderSettings.showFill) {
        ctx.fillStyle = isHovered ? hexToRgba(color, 0.35) : hexToRgba(color, 0.12);
        ctx.fillRect(b.x1, b.y1, b.width, b.height);
      }

      // 2. Bounding Box Border
      ctx.lineWidth = isHovered ? Math.max(3, w * 0.004) : Math.max(2, w * 0.002);
      ctx.strokeStyle = color;
      if (isHovered) {
        ctx.shadowColor = color;
        ctx.shadowBlur = 14;
      } else {
        ctx.shadowBlur = 0;
      }
      ctx.strokeRect(b.x1, b.y1, b.width, b.height);
      ctx.shadowBlur = 0; // reset shadow

      // 3. Label Badge Pill
      if (state.renderSettings.showLabels) {
        const confText = state.renderSettings.showConf ? ` ${(det.confidence * 100).toFixed(0)}%` : '';
        const nameStr = det.display_name || det.class_name;
        const labelText = `${nameStr}${confText}`;

        const fontSize = Math.max(12, Math.min(22, Math.round(w * 0.016)));
        ctx.font = `600 ${fontSize}px var(--font-body, 'Inter', sans-serif)`;

        const textMetrics = ctx.measureText(labelText);
        const paddingX = fontSize * 0.55;
        const paddingY = fontSize * 0.3;
        const pillWidth = textMetrics.width + paddingX * 2;
        const pillHeight = fontSize + paddingY * 2;

        let badgeX = b.x1;
        let badgeY = b.y1 - pillHeight - 2;
        if (badgeY < 0) badgeY = b.y1 + 2; // Keep within canvas bounds

        // Badge background
        ctx.fillStyle = color;
        roundRect(ctx, badgeX, badgeY, pillWidth, pillHeight, Math.max(3, fontSize * 0.25));
        ctx.fill();

        // Badge text
        ctx.fillStyle = '#05070c';
        ctx.textBaseline = 'middle';
        ctx.fillText(labelText, badgeX + paddingX, badgeY + pillHeight / 2);
      }
    });
  }

  function handleCanvasMouseMove(e) {
    if (!state.detections.length) return;

    const rect = el.displayCanvas.getBoundingClientRect();
    const scaleX = el.displayCanvas.width / rect.width;
    const scaleY = el.displayCanvas.height / rect.height;

    const mouseX = (e.clientX - rect.left) * scaleX;
    const mouseY = (e.clientY - rect.top) * scaleY;

    // Find smallest box that contains the cursor (so inner objects get priority)
    let found = null;
    let minArea = Infinity;

    for (const det of state.detections) {
      const b = det.box;
      if (mouseX >= b.x1 && mouseX <= b.x2 && mouseY >= b.y1 && mouseY <= b.y2) {
        const area = b.width * b.height;
        if (area < minArea) {
          minArea = area;
          found = det;
        }
      }
    }

    if (found) {
      state.hoveredDetectionId = found.id;
      renderCanvas();
      highlightCard(found.id);

      // Tooltip position
      el.boxTooltip.classList.remove('hidden');
      el.boxTooltip.textContent = `${found.class_name} • ${(found.confidence * 100).toFixed(1)}%`;
      el.boxTooltip.style.left = `${e.clientX - rect.left}px`;
      el.boxTooltip.style.top = `${e.clientY - rect.top}px`;
    } else {
      state.hoveredDetectionId = null;
      renderCanvas();
      clearHighlightCards();
      el.boxTooltip.classList.add('hidden');
    }
  }

  // ==========================================================================
  // Live Webcam Streaming & Real-Time Processing
  // ==========================================================================

  let isFrameInFlight = false;
  let consecutiveHttpErrors = 0;

  async function startWebcam() {
    try {
      showLoading(true, 'Accessing camera feed...');
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280, min: 640 }, height: { ideal: 720, min: 480 }, facingMode: 'user' }
      });
      state.webcamStream = stream;
      el.webcamVideo.srcObject = stream;
      
      try {
        await el.webcamVideo.play();
      } catch (playErr) {
        console.warn('Video play deferred:', playErr);
      }

      const syncDimensions = () => {
        const vw = el.webcamVideo.videoWidth || 640;
        const vh = el.webcamVideo.videoHeight || 480;
        el.webcamCanvas.width = vw;
        el.webcamCanvas.height = vh;
        offscreenCanvas.width = vw;
        offscreenCanvas.height = vh;
      };

      el.webcamVideo.onloadedmetadata = syncDimensions;
      el.webcamVideo.oncanplay = syncDimensions;
      syncDimensions();

      state.webcamActive = true;
      isFrameInFlight = false;
      consecutiveHttpErrors = 0;

      el.webcamIdleOverlay.classList.add('hidden');
      el.btnStopWebcam.disabled = false;
      showLoading(false);

      initWebSocketLive();
    } catch (err) {
      showLoading(false);
      console.error('Camera access error:', err);
      alert('Unable to access webcam: ' + err.message + '\n\nPlease check camera permissions in your browser.');
    }
  }

  function stopWebcam() {
    state.webcamActive = false;
    isFrameInFlight = false;
    if (state.webcamIntervalId) {
      clearInterval(state.webcamIntervalId);
      state.webcamIntervalId = null;
    }
    if (state.webcamStream) {
      state.webcamStream.getTracks().forEach(t => t.stop());
      state.webcamStream = null;
    }
    if (state.ws) {
      try { state.ws.close(); } catch (_) {}
      state.ws = null;
    }
    el.webcamIdleOverlay.classList.remove('hidden');
    el.btnStopWebcam.disabled = true;
    webcamCtx.clearRect(0, 0, el.webcamCanvas.width, el.webcamCanvas.height);
    el.webcamFpsHud.textContent = 'FPS: 0';
    if (el.webcamLiveHud) {
      el.webcamLiveHud.innerHTML = '<span class="hud-dot"></span> IDLE';
    }
  }

  function initWebSocketLive() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host || '127.0.0.1:8000';
    const wsUrl = `${protocol}//${host}/ws/live`;

    try {
      state.ws = new WebSocket(wsUrl);

      state.ws.onopen = () => {
        console.log('[OmniVision] WebSocket connected to', wsUrl);
        if (el.webcamLiveHud) {
          el.webcamLiveHud.innerHTML = '<span class="hud-dot live-pulse"></span> LIVE (WS)';
        }
        restartWebcamInterval();
      };

      state.ws.onmessage = e => {
        isFrameInFlight = false;
        try {
          const data = JSON.parse(e.data);
          if (data.error) {
            console.warn('[OmniVision] Live stream notice:', data.error);
            return;
          }
          if (data.detections) {
            renderWebcamDetections(data.detections);
            updateTelemetry(data.metrics, data.detections.length);
            updateDetectionsUI(data.detections, data.class_counts, data.category_counts);

            // FPS Tracker
            const now = performance.now();
            state.frameCount++;
            if (now - state.lastFrameTime >= 1000) {
              state.webcamFps = state.frameCount;
              state.frameCount = 0;
              state.lastFrameTime = now;
              el.webcamFpsHud.textContent = `FPS: ${state.webcamFps}`;
            }
          }
        } catch (err) {
          console.error('WS parse error:', err);
        }
      };

      state.ws.onerror = err => {
        console.warn('[OmniVision] WebSocket error, switching to HTTP polling:', err);
        isFrameInFlight = false;
        restartWebcamIntervalHTTP();
      };

      state.ws.onclose = () => {
        if (state.webcamActive) {
          console.log('[OmniVision] WebSocket closed, using HTTP fallback.');
          isFrameInFlight = false;
          restartWebcamIntervalHTTP();
        }
      };
    } catch (wsErr) {
      console.warn('[OmniVision] Could not create WebSocket, using HTTP fallback:', wsErr);
      restartWebcamIntervalHTTP();
    }
  }

  function restartWebcamInterval() {
    if (state.webcamIntervalId) clearInterval(state.webcamIntervalId);
    state.webcamIntervalId = setInterval(captureAndSendFrameWS, state.streamIntervalMs);
  }

  function restartWebcamIntervalHTTP() {
    if (state.webcamIntervalId) clearInterval(state.webcamIntervalId);
    if (el.webcamLiveHud) {
      el.webcamLiveHud.innerHTML = '<span class="hud-dot live-pulse" style="background:#f59e0b; box-shadow:0 0 8px #f59e0b"></span> LIVE (HTTP)';
    }
    state.webcamIntervalId = setInterval(captureAndSendFrameHTTP, state.streamIntervalMs);
  }

  const offscreenCanvas = document.createElement('canvas');
  const offscreenCtx = offscreenCanvas.getContext('2d');

  function captureAndSendFrameWS() {
    if (!state.webcamActive || !state.ws || state.ws.readyState !== WebSocket.OPEN) return;
    if (isFrameInFlight) return; // Drop frame to prevent queuing lag
    if (!el.webcamVideo.videoWidth || !el.webcamVideo.videoHeight) return;

    if (offscreenCanvas.width !== el.webcamVideo.videoWidth) {
      offscreenCanvas.width = el.webcamVideo.videoWidth;
      offscreenCanvas.height = el.webcamVideo.videoHeight;
    }

    offscreenCtx.drawImage(el.webcamVideo, 0, 0);
    const b64 = offscreenCanvas.toDataURL('image/jpeg', 0.65);
    if (!b64 || b64.length < 50) return;

    isFrameInFlight = true;
    const payload = {
      image: b64,
      confidence: state.confidence,
      iou: state.iou
    };
    state.ws.send(JSON.stringify(payload));
  }

  async function captureAndSendFrameHTTP() {
    if (!state.webcamActive) return;
    if (isFrameInFlight) return; // Drop frame to maintain high interactive FPS
    if (!el.webcamVideo.videoWidth || !el.webcamVideo.videoHeight) return;

    if (offscreenCanvas.width !== el.webcamVideo.videoWidth) {
      offscreenCanvas.width = el.webcamVideo.videoWidth;
      offscreenCanvas.height = el.webcamVideo.videoHeight;
    }

    offscreenCtx.drawImage(el.webcamVideo, 0, 0);
    const b64 = offscreenCanvas.toDataURL('image/jpeg', 0.65);
    if (!b64 || b64.length < 50) return;

    isFrameInFlight = true;
    try {
      const res = await fetch('/api/detect-base64', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image: b64,
          confidence: state.confidence,
          iou: state.iou
        })
      });
      if (res.ok) {
        consecutiveHttpErrors = 0;
        const data = await res.json();
        renderWebcamDetections(data.detections || []);
        updateTelemetry(data.metrics, (data.detections || []).length);
        updateDetectionsUI(data.detections || [], data.class_counts, data.category_counts);

        const now = performance.now();
        state.frameCount++;
        if (now - state.lastFrameTime >= 1000) {
          state.webcamFps = state.frameCount;
          state.frameCount = 0;
          state.lastFrameTime = now;
          el.webcamFpsHud.textContent = `FPS: ${state.webcamFps}`;
        }
      } else {
        consecutiveHttpErrors++;
      }
    } catch (e) {
      consecutiveHttpErrors++;
      if (consecutiveHttpErrors >= 3 && el.webcamLiveHud) {
        el.webcamLiveHud.innerHTML = '<span class="hud-dot" style="background:#f43f5e; box-shadow:0 0 8px #f43f5e"></span> SERVER OFFLINE';
      }
    } finally {
      isFrameInFlight = false;
    }
  }

  function renderWebcamDetections(detections) {
    const vw = el.webcamVideo.videoWidth || 640;
    const vh = el.webcamVideo.videoHeight || 480;

    if (el.webcamCanvas.width !== vw || el.webcamCanvas.height !== vh) {
      el.webcamCanvas.width = vw;
      el.webcamCanvas.height = vh;
    }

    const w = el.webcamCanvas.width;
    const h = el.webcamCanvas.height;
    webcamCtx.clearRect(0, 0, w, h);

    if (!state.renderSettings.showBoxes || !detections.length) return;

    detections.forEach(det => {
      const b = det.box;
      const color = det.color || '#00f0ff';

      // Mirror coordinates horizontally to match the mirrored webcam video
      const mx1 = w - b.x2;
      const mwidth = b.width;

      if (state.renderSettings.showFill) {
        webcamCtx.fillStyle = hexToRgba(color, 0.16);
        webcamCtx.fillRect(mx1, b.y1, mwidth, b.height);
      }

      // Styled corner bounding box
      webcamCtx.lineWidth = 3;
      webcamCtx.strokeStyle = color;
      webcamCtx.strokeRect(mx1, b.y1, mwidth, b.height);

      if (state.renderSettings.showLabels) {
        const confText = state.renderSettings.showConf ? ` ${(det.confidence * 100).toFixed(0)}%` : '';
        const nameStr = det.display_name || det.class_name;
        const labelText = ` ${nameStr}${confText} `;

        webcamCtx.font = '600 13px var(--font-body, "Inter", sans-serif)';
        const textMetrics = webcamCtx.measureText(labelText);
        const pillW = textMetrics.width + 10;
        const pillH = 22;
        const pillY = Math.max(0, b.y1 - pillH - 3);

        webcamCtx.fillStyle = color;
        roundRect(webcamCtx, mx1, pillY, pillW, pillH, 4);
        webcamCtx.fill();

        webcamCtx.fillStyle = '#05070c';
        webcamCtx.textBaseline = 'middle';
        webcamCtx.fillText(labelText, mx1 + 5, pillY + pillH / 2);
      }
    });
  }

  // ==========================================================================
  // Export & Utilities
  // ==========================================================================

  function exportResultsJson() {
    if (!state.detections.length) return;
    const exportData = {
      timestamp: new Date().toISOString(),
      model: state.modelInfo ? state.modelInfo.model_name : 'YOLOv8',
      confidence_threshold: state.confidence,
      iou_threshold: state.iou,
      metrics: state.metrics,
      total_detections: state.detections.length,
      detections: state.detections
    };

    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `omnivision_detections_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  function downloadAnnotatedImage() {
    if (!state.currentImageElement) return;
    const url = el.displayCanvas.toDataURL('image/png');
    const a = document.createElement('a');
    a.href = url;
    a.download = `omnivision_annotated_${Date.now()}.png`;
    a.click();
  }

  function showLoading(show, message = 'Processing...') {
    if (show) {
      el.loadingText.textContent = message;
      el.loadingOverlay.classList.remove('hidden');
    } else {
      el.loadingOverlay.classList.add('hidden');
    }
  }

  function hexToRgba(hex, alpha) {
    let c = hex.replace('#', '');
    if (c.length === 3) c = c.split('').map(x => x + x).join('');
    const r = parseInt(c.substring(0, 2), 16);
    const g = parseInt(c.substring(2, 4), 16);
    const b = parseInt(c.substring(4, 6), 16);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
  }

  function roundRect(ctx, x, y, width, height, radius) {
    ctx.beginPath();
    ctx.moveTo(x + radius, y);
    ctx.lineTo(x + width - radius, y);
    ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
    ctx.lineTo(x + width, y + height - radius);
    ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
    ctx.lineTo(x + radius, y + height);
    ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
    ctx.lineTo(x, y + radius);
    ctx.quadraticCurveTo(x, y, x + radius, y);
    ctx.closePath();
  }

  // Kickstart on DOM Load
  document.addEventListener('DOMContentLoaded', init);
})();
