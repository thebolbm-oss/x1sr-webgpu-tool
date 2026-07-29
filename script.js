/**
 * AI Image Upscaler - XLSR 3x
 * Working version with proper WASM loading
 */

(function () {
    'use strict';

    // ============ DOM ELEMENTS ============
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

    // ============ STATE ============
    let ortSession = null;
    let isModelReady = false;
    let isProcessing = false;
    let currentImage = null;
    let isMobileDevice = false;

    // ============ DEVICE DETECTION ============
    function detectMobileDevice() {
        const userAgent = navigator.userAgent || navigator.vendor || window.opera;
        const mobileRegex = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i;
        return mobileRegex.test(userAgent) || 
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

    // ============ DEBUG POPUP ============
    function showDebug(message) {
        debugContent.textContent = message;
        debugPopup.classList.remove('hidden');
    }

    function hideDebug() {
        debugPopup.classList.add('hidden');
    }

    closeDebug.addEventListener('click', hideDebug);

    // ============ STATUS BADGE ============
    function updateStatus(state, message) {
        statusDot.className = 'status-dot';
        if (state === 'loading') statusDot.classList.add('loading');
        else if (state === 'ready') statusDot.classList.add('ready');
        else if (state === 'error') statusDot.classList.add('error');
        statusText.textContent = message;
    }

    // ============ PROGRESS ============
    function updateProgress(percent, message) {
        const clampedPercent = Math.min(100, Math.max(0, Math.round(percent)));
        progressFill.style.width = clampedPercent + '%';
        progressText.textContent = clampedPercent + '%';
        if (message) processingStatus.textContent = message;
    }

    // ============ MODEL LOADING ============
    async function loadModel() {
        updateStatus('loading', 'Loading AI Model...');
        
        try {
            // Step 1: Check ONNX Runtime
            if (typeof ort === 'undefined') {
                throw new Error('ONNX Runtime not loaded. Check internet connection.');
            }
            console.log('✅ ONNX Runtime version:', ort.env?.version || 'loaded');

            // Step 2: Fetch model files
            updateProgress(20, 'Fetching model files...');
            
            const modelUrl = 'static/xlsr.onnx';
            
            // Try to load directly from path
            console.log('Loading model from:', modelUrl);
            
            const sessionOptions = {
                executionProviders: ['wasm'],  // Use WASM first for reliability
                graphOptimizationLevel: 'all',
                logSeverityLevel: 0
            };

            updateProgress(40, 'Creating inference session...');
            
            let session;
            try {
                // Direct path loading - ONNX Runtime will automatically load .data file
                session = await ort.InferenceSession.create(modelUrl, sessionOptions);
                console.log('✅ Session created via direct path');
            } catch (pathError) {
                console.warn('Direct path failed:', pathError.message);
                
                // Fallback: Fetch files manually
                updateProgress(50, 'Trying manual fetch...');
                
                const [modelRes, dataRes] = await Promise.all([
                    fetch('static/xlsr.onnx'),
                    fetch('static/xlsr.data')
                ]);

                if (!modelRes.ok || !dataRes.ok) {
                    throw new Error('Failed to fetch model files. Check if files exist in static folder.');
                }

                const modelBuffer = await modelRes.arrayBuffer();
                const dataBuffer = await dataRes.arrayBuffer();
                
                console.log('Model size:', (modelBuffer.byteLength / 1024).toFixed(1), 'KB');
                console.log('Data size:', (dataBuffer.byteLength / 1024).toFixed(1), 'KB');

                // Try with external data
                session = await ort.InferenceSession.create(modelBuffer, {
                    executionProviders: ['wasm'],
                    graphOptimizationLevel: 'all'
                });
                
                console.log('✅ Session created via manual fetch');
            }

            // Step 3: Validate session
            updateProgress(80, 'Validating model...');
            
            if (!session || !session.inputNames || session.inputNames.length === 0) {
                throw new Error('Invalid model session');
            }

            console.log('Model inputs:', session.inputNames);
            console.log('Model outputs:', session.outputNames);

            ortSession = session;
            isModelReady = true;

            updateProgress(100, 'Model ready!');
            updateStatus('ready', 'Ready');
            
            setTimeout(() => {
                processingSection.classList.add('hidden');
                uploadSection.classList.remove('hidden');
            }, 500);

        } catch (error) {
            console.error('❌ Model loading failed:', error);
            isModelReady = false;
            ortSession = null;
            updateStatus('error', 'Error');
            
            let debugMsg = 'MODEL LOADING ERROR\n\n';
            debugMsg += 'Error: ' + error.message + '\n\n';
            debugMsg += 'Troubleshooting:\n';
            debugMsg += '1. Check if these files exist:\n';
            debugMsg += '   - static/xlsr.onnx\n';
            debugMsg += '   - static/xlsr.data\n';
            debugMsg += '2. Clear browser cache (Ctrl+Shift+Del)\n';
            debugMsg += '3. Check browser console (F12)\n';
            debugMsg += '4. Use Chrome or Edge browser\n';
            debugMsg += '5. Make sure server allows .onnx files\n\n';
            debugMsg += 'Browser: ' + navigator.userAgent;
            
            showDebug(debugMsg);
            throw error;
        }
    }

    // ============ IMAGE PROCESSING ============
    function loadImageFromFile(file) {
        return new Promise((resolve, reject) => {
            if (!file || !file.type.startsWith('image/')) {
                reject(new Error('Invalid file type. Use JPG, PNG, or WebP.'));
                return;
            }

            const reader = new FileReader();
            reader.onload = (e) => {
                const img = new Image();
                img.onload = () => resolve(img);
                img.onerror = () => reject(new Error('Cannot load image. File may be corrupted.'));
                img.src = e.target.result;
            };
            reader.onerror = () => reject(new Error('Cannot read file.'));
            reader.readAsDataURL(file);
        });
    }

    function preprocessImage(image) {
        const inputSize = 128;
        const canvas = document.createElement('canvas');
        canvas.width = inputSize;
        canvas.height = inputSize;
        const ctx = canvas.getContext('2d');

        // Black background
        ctx.fillStyle = '#000000';
        ctx.fillRect(0, 0, inputSize, inputSize);

        // Scale and center
        const scale = Math.min(inputSize / image.width, inputSize / image.height);
        const w = Math.round(image.width * scale);
        const h = Math.round(image.height * scale);
        const x = Math.floor((inputSize - w) / 2);
        const y = Math.floor((inputSize - h) / 2);

        ctx.drawImage(image, x, y, w, h);

        const imageData = ctx.getImageData(0, 0, inputSize, inputSize);
        const pixels = imageData.data;

        // NCHW format: [1, 3, 128, 128]
        const floatData = new Float32Array(3 * inputSize * inputSize);
        
        for (let y = 0; y < inputSize; y++) {
            for (let x = 0; x < inputSize; x++) {
                const srcIdx = (y * inputSize + x) * 4;
                for (let c = 0; c < 3; c++) {
                    const dstIdx = c * inputSize * inputSize + y * inputSize + x;
                    floatData[dstIdx] = pixels[srcIdx + c] / 255.0;
                }
            }
        }

        return { floatData, canvas };
    }

    async function runInference(floatData) {
        if (!ortSession) {
            throw new Error('Model not loaded. Refresh the page.');
        }

        try {
            const tensor = new ort.Tensor('float32', floatData, [1, 3, 128, 128]);
            const feeds = { [ortSession.inputNames[0]]: tensor };
            const results = await ortSession.run(feeds);
            return results[ortSession.outputNames[0]];
        } catch (error) {
            throw new Error('Inference failed: ' + error.message);
        }
    }

    function postprocessOutput(outputTensor) {
        const outputSize = 384;
        const data = outputTensor.data;
        const canvas = document.createElement('canvas');
        canvas.width = outputSize;
        canvas.height = outputSize;
        const ctx = canvas.getContext('2d');
        const imageData = ctx.createImageData(outputSize, outputSize);

        for (let y = 0; y < outputSize; y++) {
            for (let x = 0; x < outputSize; x++) {
                const idx = (y * outputSize + x) * 4;
                for (let c = 0; c < 3; c++) {
                    const srcIdx = c * outputSize * outputSize + y * outputSize + x;
                    imageData.data[idx + c] = Math.min(255, Math.max(0, Math.round(data[srcIdx] * 255)));
                }
                imageData.data[idx + 3] = 255;
            }
        }

        ctx.putImageData(imageData, 0, 0);
        return canvas;
    }

    async function processImage(file) {
        if (isProcessing) return;
        isProcessing = true;
        hideDebug();

        try {
            uploadSection.classList.add('hidden');
            resultsSection.classList.add('hidden');
            processingSection.classList.remove('hidden');

            updateProgress(10, 'Loading image...');
            const image = await loadImageFromFile(file);
            
            updateProgress(30, 'Preprocessing...');
            const { floatData, canvas } = preprocessImage(image);
            
            updateProgress(50, 'Running AI...');
            const outputTensor = await runInference(floatData);
            
            updateProgress(80, 'Creating result...');
            const resultCanvas = postprocessOutput(outputTensor);

            // Display original
            originalCanvas.width = canvas.width;
            originalCanvas.height = canvas.height;
            originalCanvas.getContext('2d').drawImage(canvas, 0, 0);
            originalInfo.textContent = `Original: ${image.width}×${image.height}px`;

            // Display enhanced
            enhancedCanvas.width = resultCanvas.width;
            enhancedCanvas.height = resultCanvas.height;
            enhancedCanvas.getContext('2d').drawImage(resultCanvas, 0, 0);
            enhancedInfo.textContent = 'Enhanced: 384×384px (3×)';

            updateProgress(100, 'Complete!');
            
            setTimeout(() => {
                processingSection.classList.add('hidden');
                resultsSection.classList.remove('hidden');
            }, 300);

        } catch (error) {
            console.error(error);
            showDebug('Processing Error:\n' + error.message);
            processingSection.classList.add('hidden');
            uploadSection.classList.remove('hidden');
            alert('Error: ' + error.message);
        } finally {
            isProcessing = false;
        }
    }

    // ============ EVENT HANDLERS ============
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('drag-over');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('drag-over');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('drag-over');
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    uploadArea.addEventListener('click', () => {
        if (isModelReady) fileInput.click();
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
            e.target.value = '';
        }
    });

    function handleFile(file) {
        if (!isModelReady) {
            alert('Model is still loading. Please wait...');
            return;
        }
        processImage(file);
    }

    downloadBtn.addEventListener('click', () => {
        enhancedCanvas.toBlob((blob) => {
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'enhanced-3x.png';
            a.click();
            URL.revokeObjectURL(url);
        });
    });

    newImageBtn.addEventListener('click', () => {
        resultsSection.classList.add('hidden');
        uploadSection.classList.remove('hidden');
        originalCanvas.getContext('2d').clearRect(0, 0, originalCanvas.width, originalCanvas.height);
        enhancedCanvas.getContext('2d').clearRect(0, 0, enhancedCanvas.width, enhancedCanvas.height);
        hideDebug();
    });

    dismissBanner.addEventListener('click', () => {
        mobileBanner.classList.remove('visible');
        setTimeout(hideMobileBanner, 400);
        sessionStorage.setItem('bannerDismissed', 'true');
    });

    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') hideDebug();
    });

    // ============ INIT ============
    async function init() {
        console.log('🚀 Starting AI Image Upscaler...');
        isMobileDevice = detectMobileDevice();
        
        if (isMobileDevice && !sessionStorage.getItem('bannerDismissed')) {
            showMobileBanner();
        }

        try {
            await loadModel();
            console.log('✅ Ready to upscale images!');
        } catch (error) {
            console.error('❌ Init failed:', error);
            updateStatus('error', 'Failed');
        }
    }

    init();

})();
