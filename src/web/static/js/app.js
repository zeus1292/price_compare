/**
 * Price Compare - Frontend Application
 * Dark Theme Amazon-Style UI
 */

const API_BASE = '/api/v1';

// State
let currentQuery = '';
let currentInputType = 'text';
let loadingStartTime = null;
let loadingTimer = null;
let isSearchCollapsed = false;

// DOM Elements
const searchModule = document.getElementById('search-module');
const searchHeader = document.getElementById('search-header');
const searchBody = document.getElementById('search-body');
const searchQueryPreview = document.getElementById('search-query-preview');
const textSearchForm = document.getElementById('text-search-form');
const urlSearchForm = document.getElementById('url-search-form');
const imageSearchForm = document.getElementById('image-search-form');
const textQuery = document.getElementById('text-query');
const urlQuery = document.getElementById('url-query');
const imageQuery = document.getElementById('image-query');
const fileUploadArea = document.getElementById('file-upload-area');
const imagePreview = document.getElementById('image-preview');
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
const tabButtons = document.querySelectorAll('.tab-btn');
const tabContents = document.querySelectorAll('.tab-content');

// ========================================
// SEARCH MODULE COLLAPSE/EXPAND
// ========================================

function toggleSearchModule() {
    isSearchCollapsed = !isSearchCollapsed;

    if (isSearchCollapsed) {
        searchModule.classList.add('collapsed');
        searchHeader.classList.add('collapsed');

        // Show query preview when collapsed
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
// TAB SWITCHING
// ========================================

tabButtons.forEach(button => {
    button.addEventListener('click', (e) => {
        e.stopPropagation(); // Prevent triggering collapse
        const tab = button.dataset.tab;

        tabButtons.forEach(btn => btn.classList.remove('active'));
        tabContents.forEach(content => content.classList.remove('active'));

        button.classList.add('active');
        document.getElementById(`${tab}-tab`).classList.add('active');

        currentInputType = tab;
    });
});

// ========================================
// CONFIDENCE THRESHOLD SLIDER
// ========================================

confidenceThreshold.addEventListener('input', () => {
    thresholdValue.textContent = `${confidenceThreshold.value}%`;
});

// ========================================
// FILE UPLOAD & DRAG-DROP
// ========================================

// Drag and drop handlers
fileUploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    fileUploadArea.classList.add('dragover');
});

fileUploadArea.addEventListener('dragleave', () => {
    fileUploadArea.classList.remove('dragover');
});

fileUploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    fileUploadArea.classList.remove('dragover');

    const files = e.dataTransfer.files;
    if (files.length > 0 && files[0].type.startsWith('image/')) {
        imageQuery.files = files;
        handleImagePreview(files[0]);
    }
});

imageQuery.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        handleImagePreview(file);
    }
});

function handleImagePreview(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.innerHTML = `<img src="${e.target.result}" alt="Preview">`;
    };
    reader.readAsDataURL(file);
}

// ========================================
// SEARCH HANDLERS
// ========================================

textSearchForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const query = textQuery.value.trim();
    if (query) {
        await performSearch(query, 'text');
    }
});

urlSearchForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const query = urlQuery.value.trim();
    if (query) {
        await performSearch(query, 'url');
    }
});

imageSearchForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const file = imageQuery.files[0];
    if (file) {
        await performImageSearch(file);
    }
});

// Main search function
async function performSearch(query, inputType) {
    showLoading();
    hideError();
    hideResults();
    collapseSearchWithQuery(query);

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

    // Update counts
    resultCount.textContent = `${results.length} result${results.length !== 1 ? 's' : ''}`;
    processingTime.textContent = `${data.processing_time_ms}ms`;

    // Update query info
    const qi = data.query_info || {};
    const props = qi.extracted_properties || {};

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
    resultsGrid.innerHTML = results.map(product => createProductCard(product)).join('');

    showResults();
}

// Create product card HTML
function createProductCard(product) {
    const name = product.name || 'Unknown Product';
    const price = product.price ? formatPrice(product.price, product.currency) : 'N/A';
    const merchant = product.merchant || 'Unknown Merchant';
    const imageUrl = product.image_url;
    const sourceUrl = product.source_url;
    const confidence = product.match_confidence || 0;
    const source = product.match_source || 'database';

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

    return `
        <article class="product-card">
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

            // Clear form if new product
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
// KEYBOARD SHORTCUTS
// ========================================

document.addEventListener('keydown', (e) => {
    // Press '/' to focus search
    if (e.key === '/' && document.activeElement.tagName !== 'INPUT') {
        e.preventDefault();
        expandSearch();
        textQuery.focus();
    }

    // Press 'Escape' to collapse search
    if (e.key === 'Escape' && !isSearchCollapsed) {
        collapseSearchWithQuery(currentQuery);
    }
});

// ========================================
// INITIALIZE
// ========================================

document.addEventListener('DOMContentLoaded', () => {
    console.log('Price Compare initialized - Dark Theme');

    // Focus text input on load
    textQuery.focus();
});
