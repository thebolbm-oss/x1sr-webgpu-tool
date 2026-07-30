/**
 * AI Image Upscaler - XLSR 4x
 *
 * Fixes applied vs original:
 * 1. Actually tries WebGPU execution provider first, falls back to WASM
 *    (original only ever used 'wasm', despite the UI promising WebGPU).
 * 2. Preserves aspect ratio: instead of stretching every image into a
 *    512x512 square (which baked black letterbox bars into the output),
 *    we track the exact scale/offset used during preprocessing and crop
 *    the corresponding region back out of the model output.
 */

(function () {
    'use strict';

    // XLSR I/O size, per this model's own metadata.json: input [1,3,128,128],
    // output [1,3,512,512] -- that makes it a 4x model, not 3x.
    const MODEL_INPUT_SIZE = 128;
    const MODEL_OUTPUT_SIZE = 512; // 128 * 4
    const SCALE_FACTOR = MODEL_OUTPUT_SIZE / MODEL_INPUT_SIZE;

    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const uploadSection = document.getElementById('uploadSection');
    const processingSection = document.getElementById('processingSection');
    const resultsSection = document.getElementById('resultsSection');
    const progressFill = document.getElementById('progressFill');
    const progressText = document.getElementById('progressText');
    const processingStatus = document.getElementById('processingStatus');
    const originalCanvas = document.getElementById('originalCanvas');
    const enhancedCanvas = document.getElementById('enhancedCanvas');
    const originalInfo = document.getElementById('originalInfo');
    const enhancedInfo = document.getElementById('enhancedInfo');
    const downloadBtn = document.getElementById('downloadBtn');
    const newImageBtn = document.getElementById('newImageBtn');
    const statusDot = document.getElementById('statusDot');
    const statusText = document.getElementById('statusText');
    const debugPopup = document.getElementById('debugPopup');
    const debugContent = document.getElementById('debugContent');
    const closeDebug = document.getElementById('closeDebug');
    const mobileBanner = document.getElementById('mobileBanner');
    const dismissBanner = document.getElementById('dismissBanner');

    let ortSession = null;
    let isModelReady = false;
    let isProcessing = false;
    let isMobileDevice = false;
    let activeProvider = 'wasm';

    function detectMobileDevice() {
        const ua = navigator.userAgent || navigator.vendor || window.opera;
        return /Android|iPhone|iPad|iPod/i.test(ua) ||
               ('maxTouchPoints' in navigator && navigator.maxTouchPoints > 0 && window.innerWidth < 768);
    }

    function showMobileBanner() {
        if (isMobileDevice) {
            mobileBanner.classList.remove('hidden');
            requestAnimationFrame(() => mobileBanner.classList.add('visible'));
        }
    }

    function hideMobileBanner() {
        mobileBanner.classList.add('hidden');
        mobileBanner.classList.remove('visible');
    }

    function showDebug(msg) {
        debugContent.textContent = msg;
        debugPopup.classList.remove('hidden');
    }

    function hideDebug() {
        debugPopup.classList.add('hidden');
    }

    closeDebug.addEventListener('click', hideDebug);

    function updateStatus(state, msg) {
        statusDot.className = 'status-dot';
        if (state === 'loading') statusDot.classList.add('loading');
        else if (state === 'ready') statusDot.classList.add('ready');
        else if (state === 'error') statusDot.classList.add('error');
        statusText.textContent = msg;
    }

    function updateProgress(pct, msg) {
        pct = Math.min(100, Math.max(0, Math.round(pct)));
        progressFill.style.width = pct + '%';
        progressText.textContent = pct + '%';
        if (msg) processingStatus.textContent = msg;
    }

    function setUploadReady(ready) {
        uploadArea.classList.toggle('not-ready', !ready);
    }

    // This model ships as xlsr.onnx + xlsr.data (external weights) -- it is
    // NOT a true single-file model, despite being called "flat". Without
    // telling the runtime where xlsr.data lives, loading fails with a
    // cryptic error (e.g. "Lt[m] is not a function") because it can't
    // resolve the referenced external tensor file.
    const EXTERNAL_DATA_OPTION = {
        externalData: [
            { path: 'xlsr.data', data: 'static/xlsr.data' }
        ]
    };

    // Try WebGPU first (desktop, supported browsers), fall back to WASM.
    // This is what actually delivers on the "WebGPU" promise in the UI.
    async function createSession(sourceOrBuffer) {
        const webgpuAvailable = !isMobileDevice && 'gpu' in navigator;

        if (webgpuAvailable) {
            try {
                const session = await ort.InferenceSession.create(sourceOrBuffer, {
                    executionProviders: ['webgpu'],
                    graphOptimizationLevel: 'all',
                    ...EXTERNAL_DATA_OPTION
                });
                activeProvider = 'webgpu';
                return session;
            } catch (err) {
                console.warn('WebGPU EP failed, falling back to WASM:', err);
            }
        }

        const session = await ort.InferenceSession.create(sourceOrBuffer, {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all',
            ...EXTERNAL_DATA_OPTION
        });
        activeProvider = 'wasm';
        return session;
    }

    async function loadModel() {
        updateStatus('loading', 'Loading Model...');
        updateProgress(0, 'Checking runtime...');
        setUploadReady(false);

        if (typeof ort === 'undefined') {
            throw new Error('ONNX Runtime not loaded');
        }

        try {
            updateProgress(20, 'Loading XLSR model...');
            const session = await createSession('static/xlsr.onnx');

            ortSession = session;
            isModelReady = true;

            updateProgress(100, 'Ready!');
            updateStatus('ready', activeProvider === 'webgpu' ? 'Ready (WebGPU)' : 'Ready (CPU)');
            setUploadReady(true);

            setTimeout(() => {
                processingSection.classList.add('hidden');
                uploadSection.classList.remove('hidden');
            }, 500);

        } catch (err) {
            try {
                updateProgress(30, 'Retrying with manual fetch...');
                const resp = await fetch('static/xlsr.onnx');
                if (!resp.ok) throw new Error('HTTP ' + resp.status + ' fetching model');
                const buffer = await resp.arrayBuffer();

                const session = await createSession(buffer);

                ortSession = session;
                isModelReady = true;

                updateProgress(100, 'Ready!');
                updateStatus('ready', activeProvider === 'webgpu' ? 'Ready (WebGPU)' : 'Ready (CPU)');
                setUploadReady(true);

                setTimeout(() => {
                    processingSection.classList.add('hidden');
                    uploadSection.classList.remove('hidden');
                }, 500);

            } catch (err2) {
                updateStatus('error', 'Error');
                showDebug(
                    'MODEL ERROR\n\n' +
                    'Error: ' + err2.message + '\n\n' +
                    'FIX:\n' +
                    '1. Make sure BOTH static/xlsr.onnx AND static/xlsr.data exist\n' +
                    '   (this model has external weights, it is not truly single-file)\n' +
                    '2. Check metadata.json for the exact filenames/paths expected\n' +
                    '3. If serving locally, use a real HTTP server (not file://)'
                );
                throw err2;
            }
        }
    }

    function loadImage(file) {
        return new Promise((resolve, reject) => {
            if (!file || !file.type || !file.type.startsWith('image/')) {
                return reject(new Error('Select JPG, PNG or WebP'));
            }
            const reader = new FileReader();
            reader.onload = e => {
                const img = new Image();
                img.onload = () => resolve(img);
                img.onerror = () => reject(new Error('Invalid image'));
                img.src = e.target.result;
            };
            reader.onerror = () => reject(new Error('Read failed'));
            reader.readAsDataURL(file);
        });
    }

    /**
     * Resize into the model's fixed input square, preserving aspect ratio
     * (letterboxed with black padding), and remember exactly where the
     * real image content ended up so we can crop the padding back out
     * of the upscaled output.
     */
    function preprocess(img) {
        const size = MODEL_INPUT_SIZE;
        const canvas = document.createElement('canvas');
        canvas.width = size;
        canvas.height = size;
        const ctx = canvas.getContext('2d');

        ctx.fillStyle = '#000';
        ctx.fillRect(0, 0, size, size);

        const scale = Math.min(size / img.width, size / img.height);
        const w = Math.round(img.width * scale);
        const h = Math.round(img.height * scale);
        const offsetX = Math.floor((size - w) / 2);
        const offsetY = Math.floor((size - h) / 2);
        ctx.drawImage(img, offsetX, offsetY, w, h);

        const pixels = ctx.getImageData(0, 0, size, size).data;
        const tensor = new Float32Array(3 * size * size);

        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                const si = (y * size + x) * 4;
                for (let c = 0; c < 3; c++) {
                    tensor[c * size * size + y * size + x] = pixels[si + c] / 255.0;
                }
            }
        }

        return {
            tensor,
            canvas,
            crop: { offsetX, offsetY, w, h }
        };
    }

    /**
     * Render the raw model output, then crop out the (scaled) letterbox
     * padding so the final image matches the original aspect ratio
     * instead of always being a distorted 512x512 square.
     */
    function postprocess(outputData, crop) {
        const size = MODEL_OUTPUT_SIZE;
        const fullCanvas = document.createElement('canvas');
        fullCanvas.width = size;
        fullCanvas.height = size;
        const ctx = fullCanvas.getContext('2d');
        const imgData = ctx.createImageData(size, size);

        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                const di = (y * size + x) * 4;
                for (let c = 0; c < 3; c++) {
                    const si = c * size * size + y * size + x;
                    imgData.data[di + c] = Math.min(255, Math.max(0, Math.round(outputData[si] * 255)));
                }
                imgData.data[di + 3] = 255;
            }
        }
        ctx.putImageData(imgData, 0, 0);

        // Crop out the padded region, scaled up by the same factor.
        const cropX = Math.round(crop.offsetX * SCALE_FACTOR);
        const cropY = Math.round(crop.offsetY * SCALE_FACTOR);
        const cropW = Math.round(crop.w * SCALE_FACTOR);
        const cropH = Math.round(crop.h * SCALE_FACTOR);

        const finalCanvas = document.createElement('canvas');
        finalCanvas.width = cropW;
        finalCanvas.height = cropH;
        finalCanvas.getContext('2d').drawImage(
            fullCanvas,
            cropX, cropY, cropW, cropH,
            0, 0, cropW, cropH
        );

        return finalCanvas;
    }

    async function processImage(file) {
        if (isProcessing) return;
        isProcessing = true;
        hideDebug();

        try {
            uploadSection.classList.add('hidden');
            resultsSection.classList.add('hidden');
            processingSection.classList.remove('hidden');

            updateProgress(10, 'Loading...');
            const img = await loadImage(file);

            updateProgress(30, 'Preparing...');
            const { tensor, canvas: origCanvas, crop } = preprocess(img);

            updateProgress(50, 'Upscaling (' + (activeProvider === 'webgpu' ? 'WebGPU' : 'CPU') + ')...');
            const input = new ort.Tensor('float32', tensor, [1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE]);
            const output = await ortSession.run({ [ortSession.inputNames[0]]: input });
            const result = output[ortSession.outputNames[0]];

            updateProgress(80, 'Rendering...');
            const resultCanvas = postprocess(result.data, crop);

            originalCanvas.width = img.width;
            originalCanvas.height = img.height;
            originalCanvas.getContext('2d').drawImage(img, 0, 0);
            originalInfo.textContent = `Input: ${img.width}\u00d7${img.height}px`;

            enhancedCanvas.width = resultCanvas.width;
            enhancedCanvas.height = resultCanvas.height;
            enhancedCanvas.getContext('2d').drawImage(resultCanvas, 0, 0);
            enhancedInfo.textContent = `Enhanced: ${resultCanvas.width}\u00d7${resultCanvas.height}px (${SCALE_FACTOR}\u00d7, ${activeProvider})`;

            updateProgress(100, 'Done!');

            setTimeout(() => {
                processingSection.classList.add('hidden');
                resultsSection.classList.remove('hidden');
            }, 300);

        } catch (err) {
            showDebug('Error:\n' + err.message);
            processingSection.classList.add('hidden');
            uploadSection.classList.remove('hidden');
            alert('Failed: ' + err.message);
        } finally {
            isProcessing = false;
        }
    }

    uploadArea.addEventListener('dragover', e => {
        e.preventDefault();
        if (isModelReady) uploadArea.classList.add('drag-over');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('drag-over');
    });

    uploadArea.addEventListener('drop', e => {
        e.preventDefault();
        uploadArea.classList.remove('drag-over');
        if (isModelReady && e.dataTransfer.files[0]) processImage(e.dataTransfer.files[0]);
    });

    uploadArea.addEventListener('click', () => {
        if (isModelReady) fileInput.click();
    });

    fileInput.addEventListener('change', e => {
        if (e.target.files[0]) {
            processImage(e.target.files[0]);
            e.target.value = '';
        }
    });

    downloadBtn.addEventListener('click', () => {
        enhancedCanvas.toBlob(blob => {
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'enhanced-4x.png';
            a.click();
            URL.revokeObjectURL(url);
        });
    });

    newImageBtn.addEventListener('click', () => {
        resultsSection.classList.add('hidden');
        uploadSection.classList.remove('hidden');
        originalCanvas.width = 0;
        enhancedCanvas.width = 0;
        hideDebug();
    });

    dismissBanner.addEventListener('click', () => {
        mobileBanner.classList.remove('visible');
        setTimeout(hideMobileBanner, 400);
        sessionStorage.setItem('bannerDismissed', 'true');
    });

    document.addEventListener('keydown', e => {
        if (e.key === 'Escape') hideDebug();
    });

    async function start() {
        isMobileDevice = detectMobileDevice();
        setUploadReady(false);

        if (isMobileDevice && !sessionStorage.getItem('bannerDismissed')) {
            showMobileBanner();
        }

        try {
            await loadModel();
        } catch (err) {
            console.error(err);
        }
    }

    start();
})();
