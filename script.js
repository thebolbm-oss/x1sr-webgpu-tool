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

    // ============ DEVICE DETECTION ============
    function detectMobileDevice() {
        const userAgent = navigator.userAgent || navigator.vendor || window.opera;
        const mobileRegex = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i;
        const isMobile = mobileRegex.test(userAgent);
        // Also check for touchscreen and screen size
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
            // Small delay for animation
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

    // ============ MODEL LOADING ============
    async function loadModel() {
        updateStatus('loading', 'Loading AI Model...');
        updateProgress(10, 'Checking ONNX Runtime availability...');

        try {
            // Verify ONNX Runtime is available
            if (typeof ort === 'undefined') {
                throw new Error(
                    'ONNX Runtime Web failed to load. Please check your internet connection and ensure the CDN script is accessible.'
                );
            }

            updateProgress(20, 'Configuring execution provider...');

            // Detect best execution provider
            const isWebGPUAvailable = typeof navigator.gpu !== 'undefined';

            if (isWebGPUAvailable && !isMobileDevice) {
                // Try WebGPU on desktop first
                executionProvider = 'webgpu';
                updateProgress(30, 'Attempting WebGPU acceleration...');
            } else if (isWebGPUAvailable && isMobileDevice) {
                // On mobile with WebGPU support, still prefer WASM for stability
                executionProvider = 'wasm';
                updateProgress(30, 'Mobile detected. Using CPU mode for stability...');
            } else {
                executionProvider = 'wasm';
                updateProgress(30, 'WebGPU not available. Using CPU mode...');
            }

            // Configure session options
            const sessionOptions = {
                executionProviders: [executionProvider],
                graphOptimizationLevel: 'all',
            };

            // Add WebGPU-specific options if using WebGPU
            if (executionProvider === 'webgpu') {
                sessionOptions.preferredOutputLocation = 'gpu-buffer';
            }

            updateProgress(40, 'Fetching model files...');

            // Determine the correct base URL
            // For GitHub Pages, the base path should include the repository name if applicable
            const basePath = window.location.pathname.includes('/')
                ? window.location.pathname.substring(0, window.location.pathname.lastIndexOf('/') + 1)
                : '/';

            // Model paths - relative to the HTML file
            const modelPath = 'static/xlsr.onnx';

            updateProgress(50, 'Loading XLSR model into memory...');

            let session = null;
            let loadAttempt = 0;
            const maxAttempts = 3;

            // Try loading with retry logic
            while (loadAttempt < maxAttempts) {
                try {
                    loadAttempt++;
                    updateProgress(50 + (loadAttempt * 10), `Loading model (attempt ${loadAttempt}/${maxAttempts})...`);

                    if (executionProvider === 'webgpu') {
                        try {
                            session = await ort.InferenceSession.create(modelPath, {
                                executionProviders: ['webgpu'],
                                graphOptimizationLevel: 'all',
                            });
                            updateProgress(80, 'WebGPU model loaded successfully!');
                        } catch (webgpuError) {
                            console.warn('WebGPU loading failed, falling back to WASM:', webgpuError.message);
                            executionProvider = 'wasm';
                            updateProgress(60, 'WebGPU failed. Falling back to CPU mode...');
                            
                            // Retry with WASM
                            session = await ort.InferenceSession.create(modelPath, {
                                executionProviders: ['wasm'],
                                graphOptimizationLevel: 'all',
                            });
                            updateProgress(80, 'CPU model loaded successfully!');
                        }
                    } else {
                        session = await ort.InferenceSession.create(modelPath, {
                            executionProviders: ['wasm'],
                            graphOptimizationLevel: 'all',
                        });
                        updateProgress(80, 'CPU model loaded successfully!');
                    }

                    break; // Success - exit retry loop
                } catch (attemptError) {
                    console.error(`Loading attempt ${loadAttempt} failed:`, attemptError);
                    
                    if (loadAttempt >= maxAttempts) {
                        throw attemptError;
                    }
                    
                    // If WebGPU failed, try WASM
                    if (executionProvider === 'webgpu') {
                        console.warn('WebGPU failed entirely, switching to WASM fallback...');
                        executionProvider = 'wasm';
                        updateProgress(50, 'Switching to CPU fallback...');
                    }
                    
                    // Wait before retry
                    await new Promise(resolve => setTimeout(resolve, 1000));
                }
            }

            if (!session) {
                throw new Error('Failed to create ONNX inference session after multiple attempts.');
            }

            updateProgress(90, 'Validating model...');

            // Verify model inputs/outputs
            const inputNames = session.inputNames;
            const outputNames = session.outputNames;
            console.log('Model loaded successfully:', {
                provider: executionProvider,
                inputs: inputNames,
                outputs: outputNames,
            });

            ortSession = session;
            isModelReady = true;

            updateProgress(100, 'Model ready!');
            updateStatus('ready', `Ready (${executionProvider.toUpperCase()})`);

            // Brief delay to show 100%
            await new Promise(resolve => setTimeout(resolve, 500));

        } catch (error) {
            console.error('Model loading error:', error);
            isModelReady = false;
            ortSession = null;
            updateStatus('error', 'Error');
            updateProgress(0, '');
            
            // Show debug popup with error details
            const errorMsg = `Model Loading Failed:\n\n` +
                `Error: ${error.message}\n\n` +
                `Execution Provider: ${executionProvider}\n` +
                `Mobile Device: ${isMobileDevice}\n` +
                `WebGPU Available: ${typeof navigator.gpu !== 'undefined'}\n\n` +
                `Troubleshooting:\n` +
                `1. Ensure model files exist at: /static/xlsr.onnx and /static/xlsr.data\n` +
                `2. Check browser console for CORS errors\n` +
                `3. Try using Chrome/Edge on desktop\n` +
                `4. Ensure your server serves .onnx and .data files with correct MIME types`;
            
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
                img.onerror = () => reject(new Error('Failed to load image. The file may be corrupted.'));
                img.src = e.target.result;
            };
            reader.onerror = () => reject(new Error('Failed to read the file.'));
            reader.readAsDataURL(file);
        });
    }

    function preprocessImage(image) {
        // Resize to 128x128 (model input size for XLSR 3x)
        const inputSize = 128;
        const offscreenCanvas = document.createElement('canvas');
        offscreenCanvas.width = inputSize;
        offscreenCanvas.height = inputSize;
        const ctx = offscreenCanvas.getContext('2d');

        // Fill with black background
        ctx.fillStyle = '#000000';
        ctx.fillRect(0, 0, inputSize, inputSize);

        // Calculate aspect ratio preserving fit
        const scale = Math.min(inputSize / image.width, inputSize / image.height);
        const scaledWidth = Math.round(image.width * scale);
        const scaledHeight = Math.round(image.height * scale);
        const offsetX = Math.floor((inputSize - scaledWidth) / 2);
        const offsetY = Math.floor((inputSize - scaledHeight) / 2);

        ctx.drawImage(image, offsetX, offsetY, scaledWidth, scaledHeight);

        // Get image data
        const imageData = ctx.getImageData(0, 0, inputSize, inputSize);
        const { data } = imageData;

        // Convert to float32 RGB normalized [0, 1] with NCHW format
        const channels = 3;
        const floatData = new Float32Array(1 * channels * inputSize * inputSize);

        for (let y = 0; y < inputSize; y++) {
            for (let x = 0; x < inputSize; x++) {
                const pixelIndex = (y * inputSize + x) * 4; // RGBA
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
            throw new Error('Model not loaded. Please refresh the page and try again.');
        }

        const inputName = ortSession.inputNames[0];
        const outputName = ortSession.outputNames[0];

        // Create tensor
        const tensor = new ort.Tensor('float32', floatData, [1, 3, 128, 128]);

        // Run inference
        const feeds = { [inputName]: tensor };
        const results = await ortSession.run(feeds);
        const output = results[outputName];

        return output;
    }

    function postprocessOutput(outputTensor) {
        const outputData = outputTensor.data;
        const outputSize = 384; // 3x of 128
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
                imageData.data[pixelIndex + 3] = 255; // Alpha
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
            // Show processing UI
            uploadSection.classList.add('hidden');
            resultsSection.classList.add('hidden');
            processingSection.classList.remove('hidden');

            updateProgress(0, 'Loading image...');
            const image = await loadImageFromFile(file);
            currentImage = image;

            updateProgress(15, 'Preprocessing image...');
            const { floatData, offscreenCanvas: preprocessedCanvas } = preprocessImage(image);

            updateProgress(30, 'Running AI inference...');
            const outputTensor = await runInference(floatData);

            updateProgress(70, 'Postprocessing results...');
            const resultCanvas = postprocessOutput(outputTensor);

            // Display results
            updateProgress(85, 'Rendering output...');
            
            // Display original image
            const origCtx = originalCanvas.getContext('2d');
            originalCanvas.width = preprocessedCanvas.width;
            originalCanvas.height = preprocessedCanvas.height;
            origCtx.drawImage(preprocessedCanvas, 0, 0);
            
            // Display enhanced image
            const enhCtx = enhancedCanvas.getContext('2d');
            enhancedCanvas.width = resultCanvas.width;
            enhancedCanvas.height = resultCanvas.height;
            enhCtx.drawImage(resultCanvas, 0, 0);

            // Show image info
            originalInfo.textContent = `Input: ${image.width}×${image.height} → 128×128`;
            enhancedInfo.textContent = `Output: 384×384 (3× upscale)`;

            updateProgress(100, 'Done!');
            
            // Switch to results view
            await new Promise(resolve => setTimeout(resolve, 300));
            processingSection.classList.add('hidden');
            resultsSection.classList.remove('hidden');

        } catch (error) {
            console.error('Processing error:', error);
            
            // Show error in debug popup
            const errorMsg = `Image Processing Failed:\n\n` +
                `Error: ${error.message}\n\n` +
                `Stack: ${error.stack || 'No stack trace available'}\n\n` +
                `Model Ready: ${isModelReady}\n` +
                `Provider: ${executionProvider}`;
            
            showDebug(errorMsg);
            
            // Reset UI
            processingSection.classList.add('hidden');
            resultsSection.classList.add('hidden');
            uploadSection.classList.remove('hidden');
            updateProgress(0, '');
            
            alert('Processing failed: ' + error.message + '\n\nCheck the debug popup for details.');
        } finally {
            isProcessing = false;
        }
    }

    // ============ EVENT HANDLERS ============
    function handleFileSelect(file) {
        if (!isModelReady) {
            alert('AI model is still loading. Please wait for the status to show "Ready".');
            return;
        }
        if (file) {
            processImage(file);
        }
    }

    // Drag and drop
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

    // Click to upload
    uploadArea.addEventListener('click', () => {
        if (isModelReady) {
            fileInput.click();
        } else if (!isModelReady && ortSession === null) {
            // Model still loading
            alert('Please wait for the AI model to finish loading.');
        }
    });

    fileInput.addEventListener('change', (e) => {
        const files = e.target.files;
        if (files.length > 0) {
            handleFileSelect(files[0]);
        }
        // Reset file input so the same file can be selected again
        fileInput.value = '';
    });

    // Download button
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

    // New image button
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

    // Mobile banner dismiss
    dismissBanner.addEventListener('click', () => {
        mobileBanner.classList.remove('visible');
        setTimeout(() => {
            hideMobileBanner();
        }, 400);
        
        // Remember dismissal for this session
        sessionStorage.setItem('bannerDismissed', 'true');
    });

    // Keyboard shortcut for closing debug popup
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            hideDebug();
        }
    });

    // ============ INITIALIZATION ============
    async function initializeApp() {
        updateStatus('loading', 'Initializing...');

        // Detect mobile device
        isMobileDevice = detectMobileDevice();
        
        // Show mobile banner if applicable and not previously dismissed
        const bannerDismissed = sessionStorage.getItem('bannerDismissed');
        if (isMobileDevice && !bannerDismissed) {
            showMobileBanner();
        }

        try {
            await loadModel();
        } catch (error) {
            console.error('Application initialization failed:', error);
            
            // The debug popup is already shown by loadModel()
            // Keep the upload area visible but disabled
            updateStatus('error', 'Model Failed');
            
            // Add a helpful message above the upload area
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
            errorNotice.textContent = '⚠️ AI model failed to load. Some features are unavailable.';
            
            // Insert after header
            const header = document.querySelector('.header');
            header.insertAdjacentElement('afterend', errorNotice);
        }
    }

    // Start the application
    initializeApp().catch((error) => {
        console.error('Fatal initialization error:', error);
    });

})();