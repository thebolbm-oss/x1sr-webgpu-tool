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
    let executionProvider = 'wasm'; // Default fallback
    let isMobileDevice = false;
    let ortWasmInitialized = false;

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

    // ============ INITIALIZE ONNX RUNTIME WASM ============
    async function initializeOrtRuntime() {
        if (ortWasmInitialized) return true;
        
        try {
            updateProgress(10, 'Initializing ONNX Runtime WASM backend...');
            
            // Configure WASM path for ONNX Runtime
            ort.env.wasm.wasmPaths = {
                'ort-wasm.wasm': 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.1/dist/ort-wasm.wasm',
                'ort-wasm-simd.wasm': 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.1/dist/ort-wasm-simd.wasm',
                'ort-wasm-threaded.wasm': 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.1/dist/ort-wasm-threaded.wasm'
            };
            
            // Set WASM configuration
            ort.env.wasm.numThreads = 1; // Use single thread for better compatibility
            ort.env.wasm.simd = true; // Enable SIMD if available
            
            ortWasmInitialized = true;
            return true;
        } catch (error) {
            console.error('Failed to initialize WASM runtime:', error);
            return false;
        }
    }

    // ============ MODEL LOADING WITH MULTIPLE APPROACHES ============
    async function loadModelApproach1() {
        // Approach 1: Load from static folder with explicit external data
        updateProgress(30, 'Approach 1: Loading with external data...');
        
        try {
            const session = await ort.InferenceSession.create('static/xlsr.onnx', {
                executionProviders: [executionProvider],
                graphOptimizationLevel: 'all',
                enableCpuMemArena: true,
                enableMemPattern: true,
                logSeverityLevel: 3 // verbose logging
            });
            return session;
        } catch (error) {
            console.warn('Approach 1 failed:', error.message);
            return null;
        }
    }

    async function loadModelApproach2() {
        // Approach 2: Fetch and load from ArrayBuffer
        updateProgress(40, 'Approach 2: Loading from ArrayBuffer...');
        
        try {
            // Fetch both files manually
            const [modelResponse, dataResponse] = await Promise.all([
                fetch('static/xlsr.onnx'),
                fetch('static/xlsr.data')
            ]);
            
            if (!modelResponse.ok || !dataResponse.ok) {
                throw new Error('Failed to fetch model files');
            }
            
            const modelBuffer = await modelResponse.arrayBuffer();
            const dataBuffer = await dataResponse.arrayBuffer();
            
            // Create a combined buffer or use external data
            // ONNX Runtime Web supports loading from ArrayBuffer with external data
            const session = await ort.InferenceSession.create(modelBuffer, {
                executionProviders: [executionProvider],
                graphOptimizationLevel: 'all',
                externalData: [
                    {
                        name: 'xlsr.data',
                        data: dataBuffer
                    }
                ]
            });
            
            return session;
        } catch (error) {
            console.warn('Approach 2 failed:', error.message);
            return null;
        }
    }

    async function loadModelApproach3() {
        // Approach 3: Load from combined buffer (merge .onnx and .data)
        updateProgress(50, 'Approach 3: Loading combined model buffer...');
        
        try {
            // Fetch both files
            const [modelResponse, dataResponse] = await Promise.all([
                fetch('static/xlsr.onnx'),
                fetch('static/xlsr.data')
            ]);
            
            if (!modelResponse.ok || !dataResponse.ok) {
                throw new Error('Failed to fetch model files');
            }
            
            const modelBuffer = await modelResponse.arrayBuffer();
            const dataBuffer = await dataResponse.arrayBuffer();
            
            // Create a combined buffer
            const combinedBuffer = new Uint8Array(modelBuffer.byteLength + dataBuffer.byteLength);
            combinedBuffer.set(new Uint8Array(modelBuffer), 0);
            combinedBuffer.set(new Uint8Array(dataBuffer), modelBuffer.byteLength);
            
            const session = await ort.InferenceSession.create(combinedBuffer.buffer, {
                executionProviders: [executionProvider],
                graphOptimizationLevel: 'all'
            });
            
            return session;
        } catch (error) {
            console.warn('Approach 3 failed:', error.message);
            return null;
        }
    }

    async function loadModelApproach4() {
        // Approach 4: Try WebGPU with specific options
        if (executionProvider === 'webgpu') {
            updateProgress(60, 'Approach 4: WebGPU specific loading...');
            
            try {
                const session = await ort.InferenceSession.create('static/xlsr.onnx', {
                    executionProviders: ['webgpu'],
                    graphOptimizationLevel: 'basic', // Less optimization for better compatibility
                    enableCpuMemArena: false,
                    enableMemPattern: false
                });
                return session;
            } catch (error) {
                console.warn('Approach 4 failed:', error.message);
                return null;
            }
        }
        return null;
    }

    async function loadModelApproach5() {
        // Approach 5: WASM with minimal configuration
        updateProgress(70, 'Approach 5: WASM minimal configuration...');
        
        try {
            const session = await ort.InferenceSession.create('static/xlsr.onnx', {
                executionProviders: ['wasm'],
                graphOptimizationLevel: 'basic'
            });
            return session;
        } catch (error) {
            console.warn('Approach 5 failed:', error.message);
            return null;
        }
    }

    async function loadModel() {
        updateStatus('loading', 'Loading AI Model...');
        updateProgress(5, 'Checking ONNX Runtime availability...');

        try {
            // Verify ONNX Runtime is available
            if (typeof ort === 'undefined') {
                throw new Error(
                    'ONNX Runtime Web failed to load. Please check your internet connection.'
                );
            }

            // Initialize WASM runtime
            updateProgress(15, 'Initializing runtime...');
            await initializeOrtRuntime();

            // Detect best execution provider
            const isWebGPUAvailable = typeof navigator.gpu !== 'undefined';
            
            if (isWebGPUAvailable && !isMobileDevice) {
                executionProvider = 'webgpu';
                updateProgress(20, 'WebGPU detected. Trying GPU acceleration...');
            } else {
                executionProvider = 'wasm';
                updateProgress(20, 'Using CPU mode...');
            }

            // Log ONNX Runtime version
            console.log('ONNX Runtime version:', ort.version);
            console.log('Available execution providers:', ort.env.webgpu ? 'WebGPU available' : 'WebGPU not available');

            let session = null;
            const approaches = [];

            // Determine which approaches to try based on provider
            if (executionProvider === 'webgpu') {
                approaches.push(
                    { name: 'WebGPU External Data', fn: loadModelApproach1 },
                    { name: 'WebGPU Specific', fn: loadModelApproach4 },
                    { name: 'ArrayBuffer', fn: loadModelApproach2 }
                );
            }
            
            // Always add WASM fallback approaches
            approaches.push(
                { name: 'WASM External Data', fn: loadModelApproach5 },
                { name: 'ArrayBuffer', fn: loadModelApproach2 },
                { name: 'Combined Buffer', fn: loadModelApproach3 }
            );

            // Try each approach
            let lastError = null;
            
            for (const approach of approaches) {
                try {
                    updateProgress(25, `Trying: ${approach.name}...`);
                    session = await approach.fn();
                    
                    if (session) {
                        console.log(`✅ Model loaded successfully via: ${approach.name}`);
                        
                        // Update execution provider if we fell back
                        if (approach.name.includes('WASM')) {
                            executionProvider = 'wasm';
                        }
                        
                        break;
                    }
                } catch (error) {
                    console.warn(`❌ ${approach.name} failed:`, error.message);
                    lastError = error;
                    
                    // If WebGPU failed, switch to WASM for subsequent attempts
                    if (executionProvider === 'webgpu' && approach.name.includes('WebGPU')) {
                        console.log('WebGPU approaches failed, switching to WASM...');
                        executionProvider = 'wasm';
                        updateProgress(30, 'Falling back to CPU mode...');
                    }
                }
            }

            if (!session) {
                throw lastError || new Error('All loading approaches failed');
            }

            // Verify model
            updateProgress(90, 'Validating model...');
            console.log('Model inputs:', session.inputNames);
            console.log('Model outputs:', session.outputNames);
            
            ortSession = session;
            isModelReady = true;
            
            updateProgress(100, 'Model ready!');
            updateStatus('ready', `Ready (${executionProvider.toUpperCase()})`);
            
            await new Promise(resolve => setTimeout(resolve, 500));

        } catch (error) {
            console.error('Model loading error:', error);
            isModelReady = false;
            ortSession = null;
            updateStatus('error', 'Error');
            
            // Detailed error message
            let errorMsg = `Model Loading Failed:\n\n`;
            errorMsg += `Error: ${error.message}\n`;
            if (error.stack) {
                errorMsg += `\nStack: ${error.stack.substring(0, 300)}...\n`;
            }
            errorMsg += `\nEnvironment:\n`;
            errorMsg += `- Provider: ${executionProvider}\n`;
            errorMsg += `- Mobile: ${isMobileDevice}\n`;
            errorMsg += `- WebGPU: ${typeof navigator.gpu !== 'undefined'}\n`;
            errorMsg += `- ORT Version: ${ort?.version || 'unknown'}\n`;
            errorMsg += `\nTroubleshooting:\n`;
            errorMsg += `1. Clear browser cache and reload\n`;
            errorMsg += `2. Check console for CORS errors\n`;
            errorMsg += `3. Verify files exist: static/xlsr.onnx & static/xlsr.data\n`;
            errorMsg += `4. Try Chrome/Edge latest version\n`;
            errorMsg += `5. Check server MIME types:\n`;
            errorMsg += `   - .onnx → application/octet-stream\n`;
            errorMsg += `   - .data → application/octet-stream\n`;
            
            showDebug(errorMsg);
            throw error;
        }
    }

    // ============ IMAGE PROCESSING ============
    function loadImageFromFile(file) {
        return new Promise((resolve, reject) => {
            if (!file || !file.type.startsWith('image/')) {
                reject(new Error('Please select a valid image file.'));
                return;
            }

            const reader = new FileReader();
            reader.onload = (e) => {
                const img = new Image();
                img.onload = () => resolve(img);
                img.onerror = () => reject(new Error('Failed to load image.'));
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

        ctx.fillStyle = '#000000';
        ctx.fillRect(0, 0, inputSize, inputSize);

        const scale = Math.min(inputSize / image.width, inputSize / image.height);
        const scaledWidth = Math.round(image.width * scale);
        const scaledHeight = Math.round(image.height * scale);
        const offsetX = Math.floor((inputSize - scaledWidth) / 2);
        const offsetY = Math.floor((inputSize - scaledHeight) / 2);

        ctx.drawImage(image, offsetX, offsetY, scaledWidth, scaledHeight);

        const imageData = ctx.getImageData(0, 0, inputSize, inputSize);
        const { data } = imageData;

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

        const inputName = ortSession.inputNames[0];
        const outputName = ortSession.outputNames[0];

        const tensor = new ort.Tensor('float32', floatData, [1, 3, 128, 128]);
        const feeds = { [inputName]: tensor };
        const results = await ortSession.run(feeds);
        
        return results[outputName];
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

            updateProgress(15, 'Preprocessing...');
            const { floatData, offscreenCanvas: preprocessedCanvas } = preprocessImage(image);

            updateProgress(30, 'Running AI inference...');
            const outputTensor = await runInference(floatData);

            updateProgress(70, 'Postprocessing...');
            const resultCanvas = postprocessOutput(outputTensor);

            updateProgress(85, 'Rendering...');
            
            const origCtx = originalCanvas.getContext('2d');
            originalCanvas.width = preprocessedCanvas.width;
            originalCanvas.height = preprocessedCanvas.height;
            origCtx.drawImage(preprocessedCanvas, 0, 0);
            
            const enhCtx = enhancedCanvas.getContext('2d');
            enhancedCanvas.width = resultCanvas.width;
            enhancedCanvas.height = resultCanvas.height;
            enhCtx.drawImage(resultCanvas, 0, 0);

            originalInfo.textContent = `Input: ${image.width}×${image.height} → 128×128`;
            enhancedInfo.textContent = `Output: 384×384 (3× upscale)`;

            updateProgress(100, 'Done!');
            
            await new Promise(resolve => setTimeout(resolve, 300));
            processingSection.classList.add('hidden');
            resultsSection.classList.remove('hidden');

        } catch (error) {
            console.error('Processing error:', error);
            
            const errorMsg = `Image Processing Failed:\n\n` +
                `Error: ${error.message}\n\n` +
                `Provider: ${executionProvider}\n` +
                `Model Ready: ${isModelReady}`;
            
            showDebug(errorMsg);
            
            processingSection.classList.add('hidden');
            resultsSection.classList.add('hidden');
            uploadSection.classList.remove('hidden');
            updateProgress(0, '');
            
            alert('Processing failed. Check debug popup for details.');
        } finally {
            isProcessing = false;
        }
    }

    // ============ EVENT HANDLERS ============
    function handleFileSelect(file) {
        if (!isModelReady) {
            alert('AI model is still loading. Please wait.');
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

    uploadArea.addEventListener('click', () => {
        if (isModelReady) {
            fileInput.click();
        } else if (!isModelReady && ortSession === null) {
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
            link.download = 'enhanced-image.png';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            URL.revokeObjectURL(url);
        }, 'image/png');
    });

    newImageBtn.addEventListener('click', () => {
        resultsSection.classList.add('hidden');
        uploadSection.classList.remove('hidden');
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
        updateStatus('loading', 'Initializing...');
        isMobileDevice = detectMobileDevice();
        
        const bannerDismissed = sessionStorage.getItem('bannerDismissed');
        if (isMobileDevice && !bannerDismissed) {
            showMobileBanner();
        }

        try {
            await loadModel();
        } catch (error) {
            console.error('Initialization failed:', error);
            updateStatus('error', 'Model Failed');
            
            const errorNotice = document.createElement('div');
            errorNotice.style.cssText = `
                text-align: center;
                padding: 16px;
                margin-bottom: 16px;
                background: rgba(239, 68, 68, 0.15);
                border: 1px solid rgba(239, 68, 68, 0.3);
                border-radius: 12px;
                color: #ef4444;
                font-size: 0.9rem;
            `;
            errorNotice.textContent = '⚠️ AI model failed to load. Try refreshing the page or using Chrome/Edge.';
            
            const header = document.querySelector('.header');
            header.insertAdjacentElement('afterend', errorNotice);
        }
    }

    initializeApp().catch((error) => {
        console.error('Fatal initialization error:', error);
    });

})();