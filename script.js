/**
 * AI Image Upscaler - XLSR 3x
 * On-device AI upscaling using ONNX Runtime Web with WebGPU/WASM
 * 100% Client-Side | No Server Uploads
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
    let executionProvider = 'wasm';
    let isMobileDevice = false;

    // ============ DEVICE DETECTION ============
    function detectMobileDevice() {
        const userAgent = navigator.userAgent || navigator.vendor || window.opera;
        const mobileRegex = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i;
        const isMobile = mobileRegex.test(userAgent);
        const hasTouchScreen = (
            'maxTouchPoints' in navigator && navigator.maxTouchPoints > 0
        ) || (
            'msMaxTouchPoints' in navigator && navigator.msMaxTouchPoints > 0
        );
        const isSmallScreen = window.innerWidth < 768;
        return isMobile || (hasTouchScreen && isSmallScreen);
    }

    function showMobileBanner() {
        if (isMobileDevice) {
            mobileBanner.classList.remove('hidden');
            requestAnimationFrame(() => {
                mobileBanner.classList.add('visible');
            });
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
        if (state === 'loading') {
            statusDot.classList.add('loading');
        } else if (state === 'ready') {
            statusDot.classList.add('ready');
        } else if (state === 'error') {
            statusDot.classList.add('error');
        }
        statusText.textContent = message;
    }

    // ============ PROGRESS ============
    function updateProgress(percent, message) {
        const clampedPercent = Math.min(100, Math.max(0, Math.round(percent)));
        progressFill.style.width = clampedPercent + '%';
        progressText.textContent = clampedPercent + '%';
        if (message) {
            processingStatus.textContent = message;
        }
    }

    // ============ FETCH MODEL FILES ============
    async function fetchModelFiles() {
        updateProgress(15, 'Checking model files...');
        
        try {
            // Check if model files exist
            const modelResponse = await fetch('static/xlsr.onnx');
            const dataResponse = await fetch('static/xlsr.data');
            
            if (!modelResponse.ok) {
                throw new Error(`Failed to fetch model: HTTP ${modelResponse.status}`);
            }
            if (!dataResponse.ok) {
                throw new Error(`Failed to fetch data: HTTP ${dataResponse.status}`);
            }
            
            updateProgress(25, 'Downloading model files...');
            
            const modelBuffer = await modelResponse.arrayBuffer();
            const dataBuffer = await dataResponse.arrayBuffer();
            
            console.log('Model files fetched successfully');
            console.log('Model size:', (modelBuffer.byteLength / 1024).toFixed(2), 'KB');
            console.log('Data size:', (dataBuffer.byteLength / 1024).toFixed(2), 'KB');
            
            return { modelBuffer, dataBuffer };
            
        } catch (error) {
            console.error('Failed to fetch model files:', error);
            throw new Error(`Cannot access model files: ${error.message}. Make sure static/xlsr.onnx and static/xlsr.data exist.`);
        }
    }

    // ============ LOAD MODEL WITH DIFFERENT STRATEGIES ============
    
    async function tryLoadWithWasmPath(modelBuffer, dataBuffer) {
        updateProgress(35, 'Loading with WASM path config...');
        
        try {
            // Configure WASM paths explicitly
            ort.env.wasm.wasmPaths = {
                'ort-wasm.wasm': 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.1/dist/ort-wasm.wasm',
                'ort-wasm-simd.wasm': 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.1/dist/ort-wasm-simd.wasm',
                'ort-wasm-threaded.wasm': 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.1/dist/ort-wasm-threaded.wasm'
            };
            
            // Configure WASM settings
            ort.env.wasm.numThreads = 1;
            ort.env.wasm.simd = false; // Disable SIMD for better compatibility
            
            const session = await ort.InferenceSession.create(modelBuffer, {
                executionProviders: ['wasm'],
                graphOptimizationLevel: 'basic',
                enableCpuMemArena: false,
                enableMemPattern: false
            });
            
            return session;
        } catch (error) {
            console.warn('WASM path method failed:', error.message);
            return null;
        }
    }

    async function tryLoadWithExternalData(modelBuffer, dataBuffer) {
        updateProgress(45, 'Loading with external data...');
        
        try {
            const session = await ort.InferenceSession.create(modelBuffer, {
                executionProviders: ['wasm'],
                graphOptimizationLevel: 'basic',
                externalData: [
                    {
                        name: 'xlsr.data',
                        data: dataBuffer
                    }
                ]
            });
            
            return session;
        } catch (error) {
            console.warn('External data method failed:', error.message);
            return null;
        }
    }

    async function tryLoadAsFilePath() {
        updateProgress(55, 'Loading from file path...');
        
        try {
            const session = await ort.InferenceSession.create('static/xlsr.onnx', {
                executionProviders: ['wasm'],
                graphOptimizationLevel: 'basic'
            });
            
            return session;
        } catch (error) {
            console.warn('File path method failed:', error.message);
            return null;
        }
    }

    async function tryLoadWebGPU(modelBuffer) {
        if (typeof navigator.gpu === 'undefined') {
            return null;
        }
        
        updateProgress(65, 'Attempting WebGPU acceleration...');
        
        try {
            const session = await ort.InferenceSession.create(modelBuffer, {
                executionProviders: ['webgpu'],
                graphOptimizationLevel: 'basic',
                enableCpuMemArena: false,
                enableMemPattern: false
            });
            
            executionProvider = 'webgpu';
            return session;
        } catch (error) {
            console.warn('WebGPU method failed:', error.message);
            executionProvider = 'wasm';
            return null;
        }
    }

    async function tryLoadWithOrtConfig(modelBuffer) {
        updateProgress(75, 'Loading with custom config...');
        
        try {
            // Create session with minimal config
            const sessionOptions = {
                executionProviders: ['wasm'],
                graphOptimizationLevel: 'disabled',
                logSeverityLevel: 2
            };
            
            const session = await ort.InferenceSession.create(modelBuffer, sessionOptions);
            return session;
        } catch (error) {
            console.warn('Custom config method failed:', error.message);
            return null;
        }
    }

    async function loadModel() {
        updateStatus('loading', 'Loading AI Model...');
        updateProgress(5, 'Initializing ONNX Runtime...');

        try {
            // Verify ONNX Runtime is available
            if (typeof ort === 'undefined') {
                throw new Error('ONNX Runtime Web not loaded. Check internet connection.');
            }

            console.log('ONNX Runtime loaded, version:', ort?.version || 'unknown');

            // Fetch model files first
            const { modelBuffer, dataBuffer } = await fetchModelFiles();

            updateProgress(30, 'Loading model into memory...');

            let session = null;
            const strategies = [
                { name: 'External Data', fn: () => tryLoadWithExternalData(modelBuffer, dataBuffer) },
                { name: 'File Path', fn: () => tryLoadAsFilePath() },
                { name: 'Custom Config', fn: () => tryLoadWithOrtConfig(modelBuffer) },
                { name: 'WebGPU', fn: () => tryLoadWebGPU(modelBuffer) },
                { name: 'WASM Config', fn: () => tryLoadWithWasmPath(modelBuffer, dataBuffer) }
            ];

            for (const strategy of strategies) {
                try {
                    console.log(`Trying strategy: ${strategy.name}`);
                    session = await strategy.fn();
                    
                    if (session) {
                        console.log(`✅ Success with strategy: ${strategy.name}`);
                        break;
                    }
                } catch (error) {
                    console.warn(`❌ Strategy ${strategy.name} failed:`, error.message);
                }
            }

            if (!session) {
                throw new Error('All loading strategies failed. Model might be incompatible.');
            }

            // Validate model
            updateProgress(85, 'Validating model...');
            
            if (!session.inputNames || !session.outputNames || 
                session.inputNames.length === 0 || session.outputNames.length === 0) {
                throw new Error('Invalid model: no inputs or outputs found');
            }

            console.log('Model loaded successfully');
            console.log('Input names:', session.inputNames);
            console.log('Output names:', session.outputNames);
            console.log('Provider:', executionProvider);

            ortSession = session;
            isModelReady = true;

            updateProgress(100, 'Model ready!');
            updateStatus('ready', `Ready (${executionProvider.toUpperCase()})`);

            // Short delay to show completion
            await new Promise(resolve => setTimeout(resolve, 800));

            // Hide progress after success
            processingSection.classList.add('hidden');
            uploadSection.classList.remove('hidden');

        } catch (error) {
            console.error('Model loading failed:', error);
            isModelReady = false;
            ortSession = null;
            updateStatus('error', 'Error');
            updateProgress(0, '');

            let errorMsg = 'Model Loading Failed:\n\n';
            errorMsg += `Error: ${error.message}\n\n`;
            errorMsg += `Environment Info:\n`;
            errorMsg += `• Provider: ${executionProvider}\n`;
            errorMsg += `• Mobile: ${isMobileDevice}\n`;
            errorMsg += `• WebGPU Available: ${typeof navigator.gpu !== 'undefined'}\n`;
            errorMsg += `• ORT Version: ${ort?.version || 'unknown'}\n`;
            errorMsg += `• Browser: ${navigator.userAgent}\n\n`;
            errorMsg += `Troubleshooting Steps:\n`;
            errorMsg += `1. Clear browser cache (Ctrl+Shift+Del)\n`;
            errorMsg += `2. Try Chrome/Edge latest version\n`;
            errorMsg += `3. Check Console (F12) for errors\n`;
            errorMsg += `4. Verify these files exist:\n`;
            errorMsg += `   - static/xlsr.onnx\n`;
            errorMsg += `   - static/xlsr.data\n`;
            errorMsg += `5. Check server MIME types:\n`;
            errorMsg += `   Both should be "application/octet-stream"`;

            showDebug(errorMsg);
            throw error;
        }
    }

    // ============ IMAGE PROCESSING ============
    function loadImageFromFile(file) {
        return new Promise((resolve, reject) => {
            if (!file || !file.type.startsWith('image/')) {
                reject(new Error('Please select a valid image file (JPG, PNG, WebP).'));
                return;
            }

            const reader = new FileReader();
            reader.onload = (e) => {
                const img = new Image();
                img.onload = () => resolve(img);
                img.onerror = () => reject(new Error('Failed to load image. File might be corrupted.'));
                img.src = e.target.result;
            };
            reader.onerror = () => reject(new Error('Failed to read file.'));
            reader.readAsDataURL(file);
        });
    }

    function preprocessImage(image) {
        const inputSize = 128;
        const offscreenCanvas = document.createElement('canvas');
        offscreenCanvas.width = inputSize;
        offscreenCanvas.height = inputSize;
        const ctx = offscreenCanvas.getContext('2d');

        // Black background
        ctx.fillStyle = '#000000';
        ctx.fillRect(0, 0, inputSize, inputSize);

        // Calculate aspect ratio
        const scale = Math.min(inputSize / image.width, inputSize / image.height);
        const scaledWidth = Math.round(image.width * scale);
        const scaledHeight = Math.round(image.height * scale);
        const offsetX = Math.floor((inputSize - scaledWidth) / 2);
        const offsetY = Math.floor((inputSize - scaledHeight) / 2);

        ctx.drawImage(image, offsetX, offsetY, scaledWidth, scaledHeight);

        const imageData = ctx.getImageData(0, 0, inputSize, inputSize);
        const { data } = imageData;

        // Convert to NCHW format
        const channels = 3;
        const floatData = new Float32Array(1 * channels * inputSize * inputSize);

        for (let y = 0; y < inputSize; y++) {
            for (let x = 0; x < inputSize; x++) {
                const pixelIndex = (y * inputSize + x) * 4;
                for (let c = 0; c < channels; c++) {
                    const nchwIndex = (0 * channels * inputSize * inputSize) +
                        (c * inputSize * inputSize) +
                        (y * inputSize + x);
                    floatData[nchwIndex] = data[pixelIndex + c] / 255.0;
                }
            }
        }

        return { floatData, offscreenCanvas };
    }

    async function runInference(floatData) {
        if (!ortSession) {
            throw new Error('Model not loaded. Please refresh the page.');
        }

        try {
            const inputName = ortSession.inputNames[0];
            const outputName = ortSession.outputNames[0];

            const tensor = new ort.Tensor('float32', floatData, [1, 3, 128, 128]);
            const feeds = { [inputName]: tensor };
            
            const results = await ortSession.run(feeds);
            
            if (!results || !results[outputName]) {
                throw new Error('Inference returned no output');
            }
            
            return results[outputName];
        } catch (error) {
            console.error('Inference error:', error);
            throw new Error(`Inference failed: ${error.message}`);
        }
    }

    function postprocessOutput(outputTensor) {
        const outputData = outputTensor.data;
        const outputSize = 384;
        const channels = 3;

        const offscreenCanvas = document.createElement('canvas');
        offscreenCanvas.width = outputSize;
        offscreenCanvas.height = outputSize;
        const ctx = offscreenCanvas.getContext('2d');
        const imageData = ctx.createImageData(outputSize, outputSize);

        for (let y = 0; y < outputSize; y++) {
            for (let x = 0; x < outputSize; x++) {
                const pixelIndex = (y * outputSize + x) * 4;
                for (let c = 0; c < channels; c++) {
                    const nchwIndex = (0 * channels * outputSize * outputSize) +
                        (c * outputSize * outputSize) +
                        (y * outputSize + x);
                    const value = Math.min(255, Math.max(0, Math.round(outputData[nchwIndex] * 255)));
                    imageData.data[pixelIndex + c] = value;
                }
                imageData.data[pixelIndex + 3] = 255;
            }
        }

        ctx.putImageData(imageData, 0, 0);
        return offscreenCanvas;
    }

    async function processImage(file) {
        if (isProcessing) {
            console.warn('Already processing an image.');
            return;
        }

        isProcessing = true;
        hideDebug();

        try {
            uploadSection.classList.add('hidden');
            resultsSection.classList.add('hidden');
            processingSection.classList.remove('hidden');

            updateProgress(0, 'Loading image...');
            const image = await loadImageFromFile(file);
            currentImage = image;

            updateProgress(20, 'Preprocessing image...');
            const { floatData, offscreenCanvas: preprocessedCanvas } = preprocessImage(image);

            updateProgress(40, 'Running AI upscaling...');
            const outputTensor = await runInference(floatData);

            updateProgress(75, 'Rendering result...');
            const resultCanvas = postprocessOutput(outputTensor);

            // Display original
            const origCtx = originalCanvas.getContext('2d');
            originalCanvas.width = preprocessedCanvas.width;
            originalCanvas.height = preprocessedCanvas.height;
            origCtx.drawImage(preprocessedCanvas, 0, 0);

            // Display enhanced
            const enhCtx = enhancedCanvas.getContext('2d');
            enhancedCanvas.width = resultCanvas.width;
            enhancedCanvas.height = resultCanvas.height;
            enhCtx.drawImage(resultCanvas, 0, 0);

            originalInfo.textContent = `Input: ${image.width}×${image.height}px → 128×128px`;
            enhancedInfo.textContent = `Output: 384×384px (3× upscale)`;

            updateProgress(100, 'Complete!');

            await new Promise(resolve => setTimeout(resolve, 300));
            processingSection.classList.add('hidden');
            resultsSection.classList.remove('hidden');

        } catch (error) {
            console.error('Processing error:', error);

            const errorMsg = `Processing Failed:\n\n` +
                `Error: ${error.message}\n\n` +
                `Provider: ${executionProvider}\n` +
                `Model Ready: ${isModelReady}`;

            showDebug(errorMsg);

            processingSection.classList.add('hidden');
            resultsSection.classList.add('hidden');
            uploadSection.classList.remove('hidden');
            updateProgress(0, '');

            alert('Failed to process image. See debug popup for details.');
        } finally {
            isProcessing = false;
        }
    }

    // ============ EVENT HANDLERS ============
    function handleFileSelect(file) {
        if (!isModelReady) {
            alert('AI model is still loading. Please wait for "Ready" status.');
            return;
        }
        if (file) {
            processImage(file);
        }
    }

    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        e.stopPropagation();
        uploadArea.classList.add('drag-over');
    });

    uploadArea.addEventListener('dragleave', (e) => {
        e.preventDefault();
        e.stopPropagation();
        uploadArea.classList.remove('drag-over');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        e.stopPropagation();
        uploadArea.classList.remove('drag-over');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileSelect(files[0]);
        }
    });

    uploadArea.addEventListener('click', (e) => {
        if (isModelReady) {
            fileInput.click();
        } else if (!isModelReady) {
            alert('Please wait for the AI model to finish loading.');
        }
    });

    fileInput.addEventListener('change', (e) => {
        const files = e.target.files;
        if (files.length > 0) {
            handleFileSelect(files[0]);
        }
        fileInput.value = '';
    });

    downloadBtn.addEventListener('click', () => {
        if (!enhancedCanvas) return;
        enhancedCanvas.toBlob((blob) => {
            const url = URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = url;
            link.download = 'enhanced-3x.png';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            URL.revokeObjectURL(url);
        }, 'image/png');
    });

    newImageBtn.addEventListener('click', () => {
        resultsSection.classList.add('hidden');
        uploadSection.classList.remove('hidden');
        
        // Clear canvases
        const origCtx = originalCanvas.getContext('2d');
        origCtx.clearRect(0, 0, originalCanvas.width, originalCanvas.height);
        const enhCtx = enhancedCanvas.getContext('2d');
        enhCtx.clearRect(0, 0, enhancedCanvas.width, enhancedCanvas.height);
        
        currentImage = null;
        hideDebug();
    });

    dismissBanner.addEventListener('click', () => {
        mobileBanner.classList.remove('visible');
        setTimeout(() => {
            hideMobileBanner();
        }, 400);
        sessionStorage.setItem('bannerDismissed', 'true');
    });

    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            hideDebug();
        }
    });

    // ============ INITIALIZATION ============
    async function initializeApp() {
        console.log('Initializing AI Image Upscaler...');
        updateStatus('loading', 'Initializing...');

        // Detect device
        isMobileDevice = detectMobileDevice();
        console.log('Mobile device:', isMobileDevice);
        console.log('WebGPU available:', typeof navigator.gpu !== 'undefined');

        // Show mobile banner if needed
        const bannerDismissed = sessionStorage.getItem('bannerDismissed');
        if (isMobileDevice && !bannerDismissed) {
            showMobileBanner();
        }

        // Load model
        try {
            await loadModel();
            console.log('✅ Application initialized successfully');
        } catch (error) {
            console.error('❌ Initialization failed:', error);
            updateStatus('error', 'Model Failed');

            // Add error notice
            const errorNotice = document.createElement('div');
            errorNotice.style.cssText = `
                text-align: center;
                padding: 16px;
                margin: 16px 0;
                background: rgba(239, 68, 68, 0.15);
                border: 1px solid rgba(239, 68, 68, 0.3);
                border-radius: 12px;
                color: #ef4444;
                font-size: 0.9rem;
                line-height: 1.5;
            `;
            errorNotice.innerHTML = '⚠️ <strong>AI model failed to load</strong><br>' +
                'Try: Clearing cache • Using Chrome/Edge • Checking console (F12)';
            
            const header = document.querySelector('.header');
            header.insertAdjacentElement('afterend', errorNotice);
        }
    }

    // Start app
    initializeApp().catch((error) => {
        console.error('Fatal error:', error);
    });

})();