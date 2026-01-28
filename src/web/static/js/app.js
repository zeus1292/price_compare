/**
 * Price Compare - Frontend Application
 * Unified Input with Auto-Detection
 */

const API_BASE = '/api/v1';

// State
let currentQuery = '';
let detectedInputType = 'text';
let selectedImageFile = null;
let loadingStartTime = null;
let loadingTimer = null;
let isSearchCollapsed = false;

// Search context for feedback
let lastSearchContext = {
    query: '',
    queryType: 'text',
    traceId: null,
};

// DOM Elements
const searchModule = document.getElementById('search-module');
const searchHeader = document.getElementById('search-header');
const searchBody = document.getElementById('search-body');
const searchQueryPreview = document.getElementById('search-query-preview');
const searchForm = document.getElementById('search-form');
const searchInput = document.getElementById('search-input');
const searchBtn = document.getElementById('search-btn');
const inputTypeBadge = document.getElementById('input-type-badge');
const imageDropZone = document.getElementById('image-drop-zone');
const imageInput = document.getElementById('image-input');
const dropZoneContent = document.getElementById('drop-zone-content');
const imagePreviewContainer = document.getElementById('image-preview-container');
const imagePreview = document.getElementById('image-preview');
const removeImageBtn = document.getElementById('remove-image-btn');
const enableLiveSearch = document.getElementById('enable-live-search');
const confidenceThreshold = document.getElementById('confidence-threshold');
const thresholdValue = document.getElementById('threshold-value');
const loadingSection = document.getElementById('loading-section');
const loadingStatus = document.getElementById('loading-status');
const loadingTime = document.getElementById('loading-time');
const stepExtract = document.getElementById('step-extract');
const stepSearch = document.getElementById('step-search');
const stepRank = document.getElementById('step-rank');
const resultsSection = document.getElementById('results-section');
const resultCount = document.getElementById('result-count');
const processingTime = document.getElementById('processing-time');
const queryInfo = document.getElementById('query-info');
const resultsGrid = document.getElementById('results-grid');
const errorSection = document.getElementById('error-section');
const errorMessage = document.getElementById('error-message');

// ========================================
// INPUT TYPE AUTO-DETECTION
// ========================================

const URL_PATTERN = /^(https?:\/\/|www\.)/i;

function detectInputType(value) {
    if (!value || value.trim() === '') {
        return 'text';
    }

    if (URL_PATTERN.test(value.trim())) {
        return 'url';
    }

    return 'text';
}

function updateInputTypeBadge(type) {
    detectedInputType = type;

    // Update badge text and class
    inputTypeBadge.textContent = type.charAt(0).toUpperCase() + type.slice(1);
    inputTypeBadge.className = 'input-type-badge ' + type;

    // Update input styling
    searchInput.classList.remove('url-detected');
    if (type === 'url') {
        searchInput.classList.add('url-detected');
    }
}

// Real-time input detection
searchInput.addEventListener('input', (e) => {
    const value = e.target.value;

    // If we have an image selected, clear it when user starts typing
    if (selectedImageFile && value.trim() !== '') {
        clearSelectedImage();
    }

    const type = detectInputType(value);
    updateInputTypeBadge(type);
});

// ========================================
// IMAGE HANDLING
// ========================================

function setSelectedImage(file) {
    selectedImageFile = file;
    detectedInputType = 'image';

    // Update badge
    inputTypeBadge.textContent = 'Image';
    inputTypeBadge.className = 'input-type-badge image';

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        dropZoneContent.classList.add('hidden');
        imagePreviewContainer.classList.remove('hidden');
        imageDropZone.classList.add('has-image');
    };
    reader.readAsDataURL(file);

    // Clear text input
    searchInput.value = '';
    searchInput.placeholder = 'Image selected - click Search or enter text to search';
}

function clearSelectedImage() {
    selectedImageFile = null;
    imageInput.value = '';
    imagePreview.src = '';
    dropZoneContent.classList.remove('hidden');
    imagePreviewContainer.classList.add('hidden');
    imageDropZone.classList.remove('has-image');

    // Reset to text mode
    updateInputTypeBadge('text');
    searchInput.placeholder = 'Search by product name, paste a URL, or drop an image...';
}

// Image input change
imageInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file && file.type.startsWith('image/')) {
        setSelectedImage(file);
    }
});

// Click on drop zone to trigger file input
imageDropZone.addEventListener('click', () => {
    if (!selectedImageFile) {
        imageInput.click();
    }
});

// Remove image button
removeImageBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    clearSelectedImage();
    searchInput.focus();
});

// Drag and drop
imageDropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    imageDropZone.classList.add('dragover');
});

imageDropZone.addEventListener('dragleave', (e) => {
    e.preventDefault();
    imageDropZone.classList.remove('dragover');
});

imageDropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    imageDropZone.classList.remove('dragover');

    const files = e.dataTransfer.files;
    if (files.length > 0 && files[0].type.startsWith('image/')) {
        setSelectedImage(files[0]);
    }
});

// Also allow dropping on the entire input container
const unifiedInputContainer = document.getElementById('unified-input-container');
unifiedInputContainer.addEventListener('dragover', (e) => {
    e.preventDefault();
    imageDropZone.classList.add('dragover');
});

unifiedInputContainer.addEventListener('dragleave', (e) => {
    if (!unifiedInputContainer.contains(e.relatedTarget)) {
        imageDropZone.classList.remove('dragover');
    }
});

unifiedInputContainer.addEventListener('drop', (e) => {
    e.preventDefault();
    imageDropZone.classList.remove('dragover');

    const files = e.dataTransfer.files;
    if (files.length > 0 && files[0].type.startsWith('image/')) {
        setSelectedImage(files[0]);
    }
});

// ========================================
// SEARCH MODULE COLLAPSE/EXPAND
// ========================================

function toggleSearchModule() {
    isSearchCollapsed = !isSearchCollapsed;

    if (isSearchCollapsed) {
        searchModule.classList.add('collapsed');
        searchHeader.classList.add('collapsed');

        if (currentQuery) {
            searchQueryPreview.textContent = `"${currentQuery}"`;
            searchQueryPreview.classList.remove('hidden');
        }
    } else {
        searchModule.classList.remove('collapsed');
        searchHeader.classList.remove('collapsed');
        searchQueryPreview.classList.add('hidden');
    }
}

function collapseSearchWithQuery(query) {
    currentQuery = query;
    isSearchCollapsed = true;
    searchModule.classList.add('collapsed');
    searchHeader.classList.add('collapsed');
    searchQueryPreview.textContent = `"${query}"`;
    searchQueryPreview.classList.remove('hidden');
}

function expandSearch() {
    isSearchCollapsed = false;
    searchModule.classList.remove('collapsed');
    searchHeader.classList.remove('collapsed');
    searchQueryPreview.classList.add('hidden');
}

// ========================================
// CONFIDENCE THRESHOLD SLIDER
// ========================================

confidenceThreshold.addEventListener('input', () => {
    thresholdValue.textContent = `${confidenceThreshold.value}%`;
});

// ========================================
// UNIFIED SEARCH HANDLER
// ========================================

searchForm.addEventListener('submit', async (e) => {
    e.preventDefault();

    // Determine what to search
    if (selectedImageFile) {
        // Image search
        await performImageSearch(selectedImageFile);
    } else {
        const query = searchInput.value.trim();
        if (!query) {
            showError('Please enter a product name, URL, or upload an image');
            return;
        }

        // Auto-detect type and search
        const inputType = detectInputType(query);
        await performSearch(query, inputType);
    }
});

// Main search function (text or URL)
async function performSearch(query, inputType) {
    showLoading();
    hideError();
    hideResults();

    const displayQuery = inputType === 'url' ? 'URL: ' + new URL(query).hostname : query;
    collapseSearchWithQuery(displayQuery);

    try {
        const response = await fetch(`${API_BASE}/search`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: query,
                input_type: inputType,
                enable_live_search: enableLiveSearch.checked,
                confidence_threshold: confidenceThreshold.value / 100,
                limit: 20,
            }),
        });

        if (!response.ok) {
            throw new Error(`Search failed: ${response.statusText}`);
        }

        const data = await response.json();
        displayResults(data);

    } catch (error) {
        showError(error.message);
        expandSearch();
    } finally {
        hideLoading();
    }
}

// Image search function
async function performImageSearch(file) {
    showLoading();
    hideError();
    hideResults();
    collapseSearchWithQuery(`Image: ${file.name}`);

    try {
        const formData = new FormData();
        formData.append('image', file);
        formData.append('enable_live_search', enableLiveSearch.checked);
        formData.append('confidence_threshold', confidenceThreshold.value / 100);
        formData.append('limit', 20);

        const response = await fetch(`${API_BASE}/search/image`, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            throw new Error(`Search failed: ${response.statusText}`);
        }

        const data = await response.json();
        displayResults(data);

    } catch (error) {
        showError(error.message);
        expandSearch();
    } finally {
        hideLoading();
    }
}

// ========================================
// DISPLAY RESULTS
// ========================================

function displayResults(data) {
    const results = data.results || [];

    // Store search context for feedback
    const qi = data.query_info || {};
    const props = qi.extracted_properties || {};
    lastSearchContext = {
        query: props.name || currentQuery,
        queryType: detectedInputType,
        traceId: data.trace_id || null,
    };

    // Update counts
    resultCount.textContent = `${results.length} result${results.length !== 1 ? 's' : ''}`;
    processingTime.textContent = `${data.processing_time_ms}ms`;

    let queryInfoHtml = '<div class="query-info-row">';

    if (props.name) {
        queryInfoHtml += `
            <div class="query-info-item">
                <span class="query-info-label">Query:</span>
                <span class="query-info-value">${escapeHtml(props.name)}</span>
            </div>`;
    }

    if (props.brand) {
        queryInfoHtml += `
            <div class="query-info-item">
                <span class="query-info-label">Brand:</span>
                <span class="query-info-value">${escapeHtml(props.brand)}</span>
            </div>`;
    }

    if (props.category) {
        queryInfoHtml += `
            <div class="query-info-item">
                <span class="query-info-label">Category:</span>
                <span class="query-info-value">${escapeHtml(props.category)}</span>
            </div>`;
    }

    queryInfoHtml += `
        <div class="query-info-item">
            <span class="query-info-label">Method:</span>
            <span class="query-info-value">${qi.search_method || 'N/A'}</span>
        </div>`;

    if (qi.live_search_triggered) {
        queryInfoHtml += `
            <div class="query-info-item">
                <span class="query-info-value live-search">Live search triggered</span>
            </div>`;
    }

    queryInfoHtml += '</div>';

    // Add trace link if available
    if (data.trace_id) {
        const traceUrl = `https://smith.langchain.com/o/default/projects/p/price-compare/r/${data.trace_id}`;
        queryInfoHtml += `<a href="${traceUrl}" target="_blank" class="trace-link">View trace in LangSmith</a>`;
    }

    queryInfo.innerHTML = queryInfoHtml;

    // Display results grid
    resultsGrid.innerHTML = results.map((product, index) => createProductCard(product, index)).join('');

    showResults();
}

// Create product card HTML
function createProductCard(product, index) {
    const name = product.name || 'Unknown Product';
    const price = product.price ? formatPrice(product.price, product.currency) : 'N/A';
    const merchant = product.merchant || 'Unknown Merchant';
    const imageUrl = product.image_url;
    const sourceUrl = product.source_url;
    const confidence = product.match_confidence || 0;
    const source = product.match_source || 'database';
    const productId = product.id || product.product_id || null;

    const confidenceClass = confidence >= 0.8 ? 'confidence-high' :
                           confidence >= 0.5 ? 'confidence-medium' : 'confidence-low';

    // Generate placeholder with product initials if no image
    const initials = name.split(' ').slice(0, 2).map(w => w[0]).join('').toUpperCase();
    const placeholderColor = stringToColor(name);

    const badgeClass = source === 'live' ? 'live' : 'db';
    const badgeText = source === 'live' ? 'Live' : 'DB';

    const imageHtml = imageUrl
        ? `<img src="${imageUrl}" alt="${escapeHtml(name)}" class="product-image" onerror="this.outerHTML='<div class=\\'product-image placeholder\\' style=\\'background: ${placeholderColor}\\'>${initials}</div>'">`
        : `<div class="product-image placeholder" style="background: ${placeholderColor}">${initials}</div>`;

    // Encode product data for feedback
    const feedbackData = encodeURIComponent(JSON.stringify({
        productId: productId,
        name: name,
        merchant: merchant,
        confidence: confidence,
    }));

    return `
        <article class="product-card" data-product-index="${index}">
            <div class="product-image-container">
                ${imageHtml}
                <span class="product-badge ${badgeClass}">${badgeText}</span>
            </div>
            <div class="product-info">
                <h3 class="product-name">${escapeHtml(name)}</h3>
                <p class="product-price">${price}</p>
                <p class="product-merchant">${escapeHtml(merchant)}</p>
                <div class="product-footer">
                    <span class="confidence-badge ${confidenceClass}">
                        ${(confidence * 100).toFixed(0)}% match
                    </span>
                    ${sourceUrl ? `<a href="${sourceUrl}" target="_blank" class="product-link">View Product</a>` : ''}
                </div>
                <div class="product-feedback" data-feedback="${feedbackData}">
                    <span class="feedback-label">Was this helpful?</span>
                    <button class="feedback-btn thumbs-up" onclick="submitFeedback(this, 1)" title="Good match">
                        <span class="feedback-icon">👍</span>
                    </button>
                    <button class="feedback-btn thumbs-down" onclick="submitFeedback(this, -1)" title="Poor match">
                        <span class="feedback-icon">👎</span>
                    </button>
                    <span class="feedback-status"></span>
                </div>
            </div>
        </article>
    `;
}

// Generate a consistent color from a string
function stringToColor(str) {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
        hash = str.charCodeAt(i) + ((hash << 5) - hash);
    }
    const hue = hash % 360;
    return `linear-gradient(135deg, hsl(${hue}, 60%, 45%), hsl(${(hue + 40) % 360}, 50%, 35%))`;
}

// ========================================
// UTILITY FUNCTIONS
// ========================================

function formatPrice(price, currency = 'USD') {
    const symbols = {
        'USD': '$',
        'EUR': '€',
        'GBP': '£',
        'JPY': '¥',
        'SEK': 'kr',
    };
    const symbol = symbols[currency] || currency + ' ';
    return `${symbol}${price.toFixed(2)}`;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ========================================
// LOADING STATE
// ========================================

function showLoading() {
    loadingSection.classList.remove('hidden');
    loadingStartTime = Date.now();

    // Reset steps
    [stepExtract, stepSearch, stepRank].forEach(step => {
        step.classList.remove('active', 'complete');
    });

    // Start step animation
    updateLoadingStep(1);

    // Start timer
    loadingTimer = setInterval(() => {
        const elapsed = ((Date.now() - loadingStartTime) / 1000).toFixed(1);
        loadingTime.textContent = `Elapsed: ${elapsed}s`;
    }, 100);
}

function hideLoading() {
    loadingSection.classList.add('hidden');
    if (loadingTimer) {
        clearInterval(loadingTimer);
        loadingTimer = null;
    }
}

function updateLoadingStep(step) {
    const steps = [
        { el: stepExtract, status: 'Analyzing your query...' },
        { el: stepSearch, status: 'Searching product database...' },
        { el: stepRank, status: 'Ranking and filtering results...' }
    ];

    steps.forEach((s, i) => {
        if (i < step - 1) {
            s.el.classList.remove('active');
            s.el.classList.add('complete');
        } else if (i === step - 1) {
            s.el.classList.add('active');
            s.el.classList.remove('complete');
            loadingStatus.textContent = s.status;
        } else {
            s.el.classList.remove('active', 'complete');
        }
    });

    // Auto-advance steps for demo
    if (step < 3) {
        setTimeout(() => updateLoadingStep(step + 1), 1500);
    }
}

// ========================================
// SHOW/HIDE SECTIONS
// ========================================

function showResults() {
    resultsSection.classList.remove('hidden');
}

function hideResults() {
    resultsSection.classList.add('hidden');
}

function showError(message) {
    errorMessage.textContent = message;
    errorSection.classList.remove('hidden');
}

function hideError() {
    errorSection.classList.add('hidden');
}

// ========================================
// CONTRIBUTE FORM
// ========================================

function toggleContributeForm() {
    const form = document.getElementById('contribute-form');
    form.classList.toggle('hidden');
}

async function submitProduct(event) {
    event.preventDefault();

    const resultDiv = document.getElementById('contribute-result');
    resultDiv.classList.add('hidden');

    const data = {
        name: document.getElementById('contrib-name').value.trim(),
        brand: document.getElementById('contrib-brand').value.trim() || null,
        price: parseFloat(document.getElementById('contrib-price').value) || null,
        merchant: document.getElementById('contrib-merchant').value.trim() || null,
        source_url: document.getElementById('contrib-url').value.trim() || null,
        image_url: document.getElementById('contrib-image').value.trim() || null,
        category: document.getElementById('contrib-category').value.trim() || null,
    };

    try {
        const response = await fetch(`${API_BASE}/products/contribute`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data),
        });

        const result = await response.json();

        if (response.ok) {
            resultDiv.className = 'contribute-result success';
            resultDiv.textContent = result.created
                ? `Product "${data.name}" added successfully!`
                : `Product already exists in the database.`;
            resultDiv.classList.remove('hidden');

            if (result.created) {
                document.getElementById('product-contribute-form').reset();
            }
        } else {
            throw new Error(result.detail || 'Failed to add product');
        }
    } catch (error) {
        resultDiv.className = 'contribute-result error';
        resultDiv.textContent = `Error: ${error.message}`;
        resultDiv.classList.remove('hidden');
    }
}

// ========================================
// FEEDBACK SUBMISSION
// ========================================

async function submitFeedback(button, rating) {
    const feedbackContainer = button.closest('.product-feedback');
    const statusEl = feedbackContainer.querySelector('.feedback-status');
    const productData = JSON.parse(decodeURIComponent(feedbackContainer.dataset.feedback));

    // Disable buttons while submitting
    const buttons = feedbackContainer.querySelectorAll('.feedback-btn');
    buttons.forEach(btn => btn.disabled = true);

    // Show loading
    statusEl.textContent = '...';
    statusEl.className = 'feedback-status loading';

    try {
        const response = await fetch(`${API_BASE}/feedback`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: lastSearchContext.query,
                query_type: lastSearchContext.queryType,
                rating: rating,
                trace_id: lastSearchContext.traceId,
                result_product_id: productData.productId,
                result_name: productData.name,
                result_merchant: productData.merchant,
                result_confidence: productData.confidence,
            }),
        });

        if (!response.ok) {
            throw new Error('Failed to submit feedback');
        }

        // Show success
        statusEl.textContent = rating === 1 ? 'Thanks!' : 'Got it!';
        statusEl.className = 'feedback-status success';

        // Highlight the selected button
        buttons.forEach(btn => btn.classList.remove('selected'));
        button.classList.add('selected');

        // Keep buttons disabled to prevent duplicate submissions
    } catch (error) {
        console.error('Feedback error:', error);
        statusEl.textContent = 'Error';
        statusEl.className = 'feedback-status error';

        // Re-enable buttons on error
        buttons.forEach(btn => btn.disabled = false);
    }
}

// ========================================
// KEYBOARD SHORTCUTS
// ========================================

document.addEventListener('keydown', (e) => {
    // Press '/' to focus search
    if (e.key === '/' && document.activeElement.tagName !== 'INPUT') {
        e.preventDefault();
        expandSearch();
        searchInput.focus();
    }

    // Press 'Escape' to collapse search or clear image
    if (e.key === 'Escape') {
        if (selectedImageFile) {
            clearSelectedImage();
        } else if (!isSearchCollapsed) {
            collapseSearchWithQuery(currentQuery);
        }
    }
});

// ========================================
// PASTE HANDLER (for URLs and images)
// ========================================

searchInput.addEventListener('paste', (e) => {
    // Check if pasting an image from clipboard
    const items = e.clipboardData?.items;
    if (items) {
        for (const item of items) {
            if (item.type.startsWith('image/')) {
                e.preventDefault();
                const file = item.getAsFile();
                if (file) {
                    setSelectedImage(file);
                }
                return;
            }
        }
    }

    // For text paste, let it happen naturally and detect on next tick
    setTimeout(() => {
        const type = detectInputType(searchInput.value);
        updateInputTypeBadge(type);
    }, 0);
});

// ========================================
// INITIALIZE
// ========================================

document.addEventListener('DOMContentLoaded', () => {
    console.log('Price Compare initialized - Unified Input');

    // Focus text input on load
    searchInput.focus();

    // Initialize badge
    updateInputTypeBadge('text');
});
