/**
 * Retail Right - Smart Product Price Comparison
 * Light theme with auth, trending categories, and recent searches
 */

const API_BASE = '/api/v1';

// State
let currentQuery = '';
let detectedInputType = 'text';
let selectedImageFile = null;
let loadingStartTime = null;
let loadingTimer = null;
let currentUser = null;
let lastSearchResults = [];
let currentSortBy = 'relevance';
let isSearchMode = false;

// Search context for feedback
let lastSearchContext = {
    query: '',
    queryType: 'text',
    traceId: null,
};

// DOM Elements
const searchForm = document.getElementById('search-form');
const searchInput = document.getElementById('search-input');
const searchBtn = document.getElementById('search-btn');
const cameraBtn = document.getElementById('camera-btn');
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
const resultsGrid = document.getElementById('results-grid');
const errorSection = document.getElementById('error-section');
const errorMessage = document.getElementById('error-message');
const sortBySelect = document.getElementById('sort-by');

// Auth DOM Elements
const headerActions = document.getElementById('header-actions');
const userMenu = document.getElementById('user-menu');
const userName = document.getElementById('user-name');
const authModal = document.getElementById('auth-modal');
const authModalTitle = document.getElementById('auth-modal-title');
const authForm = document.getElementById('auth-form');
const authName = document.getElementById('auth-name');
const authEmail = document.getElementById('auth-email');
const authPassword = document.getElementById('auth-password');
const authError = document.getElementById('auth-error');
const authSubmitBtn = document.getElementById('auth-submit-btn');
const nameGroup = document.getElementById('name-group');
const authSwitchText = document.getElementById('auth-switch-text');
const authSwitchBtn = document.getElementById('auth-switch-btn');

// Recent Searches DOM Elements
const recentSearchesSection = document.getElementById('recent-searches-section');
const recentSearchesList = document.getElementById('recent-searches-list');

// Popular Searches DOM Elements
const popularSearches = document.getElementById('popular-searches');

// Auth mode: 'login' or 'signup'
let authMode = 'login';

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

// ========================================
// IMAGE HANDLING
// ========================================

function setSelectedImage(file) {
    selectedImageFile = file;
    detectedInputType = 'image';

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        dropZoneContent.classList.add('hidden');
        imagePreviewContainer.classList.remove('hidden');
        imageDropZone.classList.add('has-image');
    };
    reader.readAsDataURL(file);

    // Update placeholder
    searchInput.value = '';
    searchInput.placeholder = 'Image selected - click Search';
}

function clearSelectedImage() {
    selectedImageFile = null;
    imageInput.value = '';
    imagePreview.src = '';
    dropZoneContent.classList.remove('hidden');
    imagePreviewContainer.classList.add('hidden');
    imageDropZone.classList.remove('has-image');
    imageDropZone.classList.remove('active');

    detectedInputType = 'text';
    searchInput.placeholder = 'Search for any product...';
}

// Camera button click - toggle image drop zone
cameraBtn.addEventListener('click', () => {
    if (imageDropZone.classList.contains('active') || imageDropZone.classList.contains('has-image')) {
        clearSelectedImage();
    } else {
        imageDropZone.classList.add('active');
        imageInput.click();
    }
});

// Image input change
imageInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file && file.type.startsWith('image/')) {
        setSelectedImage(file);
    }
});

// Click on drop zone to trigger file input
imageDropZone.addEventListener('click', (e) => {
    if (!selectedImageFile && e.target !== removeImageBtn) {
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

// ========================================
// CONFIDENCE THRESHOLD SLIDER
// ========================================

confidenceThreshold.addEventListener('input', () => {
    thresholdValue.textContent = `${confidenceThreshold.value}%`;
});

// ========================================
// SEARCH HANDLER
// ========================================

searchForm.addEventListener('submit', async (e) => {
    e.preventDefault();

    if (selectedImageFile) {
        await performImageSearch(selectedImageFile);
    } else {
        const query = searchInput.value.trim();
        if (!query) {
            showError('Please enter a product name or upload an image');
            return;
        }

        const inputType = detectInputType(query);
        currentQuery = query;
        await performSearch(query, inputType);
    }
});

// Main search function (text or URL)
async function performSearch(query, inputType) {
    // Update current query state
    currentQuery = query;
    detectedInputType = inputType;

    // Clear previous results state
    lastSearchResults = [];
    clearResultsHeader();

    enterSearchMode();
    showLoading();
    hideError();
    hideResults();

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
                sort_by: currentSortBy,
            }),
        });

        if (!response.ok) {
            throw new Error(`Search failed: ${response.statusText}`);
        }

        const data = await response.json();
        lastSearchResults = data.results || [];
        displayResults(data);

        // Refresh recent searches if logged in
        if (currentUser) {
            loadRecentSearches();
        }

    } catch (error) {
        showError(error.message);
    } finally {
        hideLoading();
    }
}

// Image search function
async function performImageSearch(file) {
    // Update current query state
    currentQuery = `Image: ${file.name}`;
    detectedInputType = 'image';

    // Clear previous results state
    lastSearchResults = [];
    clearResultsHeader();

    enterSearchMode();
    showLoading();
    hideError();
    hideResults();

    try {
        const formData = new FormData();
        formData.append('image', file);
        formData.append('enable_live_search', enableLiveSearch.checked);
        formData.append('confidence_threshold', confidenceThreshold.value / 100);
        formData.append('limit', 20);
        formData.append('sort_by', currentSortBy);

        const response = await fetch(`${API_BASE}/search/image`, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            throw new Error(`Search failed: ${response.statusText}`);
        }

        const data = await response.json();
        lastSearchResults = data.results || [];
        displayResults(data);

        // Refresh recent searches if logged in
        if (currentUser) {
            loadRecentSearches();
        }

    } catch (error) {
        showError(error.message);
    } finally {
        hideLoading();
    }
}

// ========================================
// SORT FUNCTIONALITY
// ========================================

function handleSortChange() {
    currentSortBy = sortBySelect.value;

    // Re-sort existing results client-side
    if (lastSearchResults.length > 0) {
        const sorted = sortResults(lastSearchResults, currentSortBy);
        resultsGrid.innerHTML = sorted.map((product, index) => createProductCard(product, index)).join('');
    }
}

function sortResults(results, sortBy) {
    const sorted = [...results];
    if (sortBy === 'price_low_high') {
        sorted.sort((a, b) => (a.price || Infinity) - (b.price || Infinity));
    } else if (sortBy === 'price_high_low') {
        sorted.sort((a, b) => (b.price || 0) - (a.price || 0));
    } else {
        // Default: relevance (by confidence)
        sorted.sort((a, b) => (b.match_confidence || 0) - (a.match_confidence || 0));
    }
    return sorted;
}

window.handleSortChange = handleSortChange;

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

    // Get display query
    const displayQuery = props.name || currentQuery;
    const truncatedQuery = displayQuery.length > 50
        ? displayQuery.substring(0, 50) + '...'
        : displayQuery;

    // Display results grid or show error
    if (results.length > 0) {
        // Update results title
        const titleText = document.getElementById('results-title-text');
        if (titleText) {
            titleText.textContent = `Results for "${truncatedQuery}"`;
        }

        // Update counts
        resultCount.textContent = `${results.length} result${results.length !== 1 ? 's' : ''}`;

        resultsGrid.innerHTML = results.map((product, index) => createProductCard(product, index)).join('');
        showResults();
    } else {
        // No results - go back home and show error message
        goHome();
        showError(`No products found for "${truncatedQuery}". Try enabling live web search or use a different search term.`);
    }
}

// Clear results header (reset title)
function clearResultsHeader() {
    const titleText = document.getElementById('results-title-text');
    if (titleText) {
        titleText.textContent = 'Searching...';
    }
}

// Validate if image URL is from a legitimate retail/CDN source
function isValidRetailImageUrl(imageUrl, sourceUrl) {
    if (!imageUrl || typeof imageUrl !== 'string') {
        return false;
    }

    // Must be a valid HTTP/HTTPS URL
    if (!imageUrl.startsWith('http://') && !imageUrl.startsWith('https://')) {
        return false;
    }

    // Must have a valid image extension or be from known image CDNs
    const validImageExtensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg', '.avif'];
    const urlLower = imageUrl.toLowerCase();
    const hasValidExtension = validImageExtensions.some(ext => urlLower.includes(ext));

    // Known retail domains and CDNs that serve product images
    const trustedDomains = [
        // Major retailers
        'amazon.com', 'amazonaws.com', 'media-amazon.com',
        'walmart.com', 'walmartimages.com',
        'target.com', 'scene7.com',
        'bestbuy.com', 'bestbuy.ca',
        'ebay.com', 'ebayimg.com',
        'etsy.com', 'etsystatic.com',
        'shopify.com', 'shopifycdn.com', 'cdn.shopify.com',
        'aliexpress.com', 'alicdn.com',
        'wayfair.com', 'wfcdn.com',
        'homedepot.com', 'thdstatic.com',
        'lowes.com', 'lowe.com',
        'costco.com',
        'macys.com',
        'nordstrom.com',
        'zappos.com',
        'overstock.com',
        'chewy.com',
        'newegg.com',
        'bhphotovideo.com',
        // CDNs commonly used by retailers
        'cloudinary.com', 'cloudfront.net', 'akamaized.net',
        'fastly.net', 'imgix.net', 'cloudflare.com',
        // Image hosting
        'googleusercontent.com', 'gstatic.com',
        'media.githubusercontent.com',
        // Regional retailers
        'johnlewis.com', 'argos.co.uk', 'currys.co.uk',
        'mediamarkt.', 'saturn.de',
        'fnac.com', 'darty.com',
        'elgiganten.', 'elkjop.',
        // Brand sites
        'apple.com', 'samsung.com', 'sony.com', 'lg.com',
        'nike.com', 'adidas.com', 'puma.com',
    ];

    try {
        const imageUrlObj = new URL(imageUrl);
        const imageHost = imageUrlObj.hostname.toLowerCase();

        // Check if image is from a trusted domain
        const isFromTrustedDomain = trustedDomains.some(domain => imageHost.includes(domain));

        // If we have a source URL, check if image is from the same domain
        let isFromSameSourceDomain = false;
        if (sourceUrl) {
            try {
                const sourceUrlObj = new URL(sourceUrl);
                const sourceHost = sourceUrlObj.hostname.toLowerCase().replace('www.', '');
                const imageHostClean = imageHost.replace('www.', '');
                // Check if they share the same root domain
                isFromSameSourceDomain = imageHostClean.includes(sourceHost) ||
                                         sourceHost.includes(imageHostClean.split('.').slice(-2).join('.'));
            } catch (e) {
                // Invalid source URL, ignore
            }
        }

        // Accept if from trusted domain, same source domain, or has valid image extension
        return isFromTrustedDomain || isFromSameSourceDomain || hasValidExtension;
    } catch (e) {
        // Invalid URL
        return false;
    }
}

// Create product card HTML
function createProductCard(product, index) {
    const name = product.name || 'Unknown Product';
    const price = product.price ? formatPrice(product.price, product.currency) : 'Price N/A';
    const merchant = product.merchant || 'Unknown';
    const brand = extractBrand(name, product.brand);
    const imageUrl = product.image_url;
    const sourceUrl = product.source_url;
    const confidence = product.match_confidence || 0;
    const productId = product.id || product.product_id || null;

    // Confidence level
    const confidencePercent = Math.round(confidence * 100);
    const confidenceClass = confidence >= 0.8 ? '' : confidence >= 0.5 ? 'medium' : 'low';

    // Generate placeholder with gradient
    const gradientColors = getGradientForString(name);

    // Encode product data for feedback
    const feedbackData = encodeURIComponent(JSON.stringify({
        productId: productId,
        name: name,
        merchant: merchant,
        confidence: confidence,
    }));

    // Only show image if it's from a valid retail source
    const hasValidImage = isValidRetailImageUrl(imageUrl, sourceUrl);
    const imageHtml = hasValidImage
        ? `<img src="${imageUrl}" alt="${escapeHtml(name)}" class="product-image" onerror="this.outerHTML='<div class=\\'product-image placeholder\\' style=\\'background: ${gradientColors}\\'>${getInitials(name)}</div>'">`
        : `<div class="product-image placeholder" style="background: ${gradientColors}">${getInitials(name)}</div>`;

    return `
        <article class="product-card" data-product-index="${index}">
            <div class="product-image-container">
                ${imageHtml}
                <div class="product-badges">
                    <div class="confidence-pill">
                        <span class="confidence-dot ${confidenceClass}"></span>
                        <span>${confidencePercent}%</span>
                    </div>
                    <div class="merchant-pill">${escapeHtml(merchant)}</div>
                </div>
            </div>
            <div class="product-info">
                <div class="product-brand">${escapeHtml(brand)}</div>
                <h3 class="product-name">${escapeHtml(name)}</h3>
                <div class="product-price-row">
                    <span class="product-price">${price}</span>
                    <div class="product-feedback-mini" data-feedback="${feedbackData}">
                        <button class="feedback-btn-mini thumbs-up" onclick="submitFeedback(this, 1)" title="Good match">👍</button>
                        <button class="feedback-btn-mini thumbs-down" onclick="submitFeedback(this, -1)" title="Poor match">👎</button>
                    </div>
                </div>
                <p class="product-description">${generateDescription(name, brand, merchant)}</p>
                ${sourceUrl
                    ? `<a href="${sourceUrl}" target="_blank" class="product-cta">
                        <span class="product-cta-icon">🛒</span>
                        Get it on ${escapeHtml(merchant)}
                       </a>`
                    : `<button class="product-cta" disabled>
                        <span class="product-cta-icon">🛒</span>
                        View on ${escapeHtml(merchant)}
                       </button>`
                }
            </div>
        </article>
    `;
}

// Extract brand from product name
function extractBrand(name, explicitBrand) {
    if (explicitBrand) return explicitBrand;

    // Common brand patterns - first word is often the brand
    const words = name.split(' ');
    if (words.length > 0) {
        const firstWord = words[0];
        // If first word looks like a brand (capitalized, not a common word)
        const commonWords = ['the', 'a', 'an', 'new', 'used', 'vintage', 'original'];
        if (!commonWords.includes(firstWord.toLowerCase()) && firstWord.length > 1) {
            return firstWord;
        }
    }
    return 'Product';
}

// Generate a short description
function generateDescription(name, brand, merchant) {
    const descriptors = [
        `Available at ${merchant}`,
        `Shop ${brand} products`,
        `Find great deals on ${brand}`,
    ];
    return descriptors[Math.floor(Math.random() * descriptors.length)];
}

// Get initials for placeholder
function getInitials(name) {
    return name.split(' ').slice(0, 2).map(w => w[0]).join('').toUpperCase();
}

// Generate gradient for placeholder
function getGradientForString(str) {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
        hash = str.charCodeAt(i) + ((hash << 5) - hash);
    }
    const hue = Math.abs(hash % 360);
    return `linear-gradient(135deg, hsl(${hue}, 70%, 60%), hsl(${(hue + 40) % 360}, 60%, 50%))`;
}

// ========================================
// FEEDBACK SUBMISSION
// ========================================

async function submitFeedback(button, rating) {
    const feedbackContainer = button.closest('.product-feedback-mini');
    const productData = JSON.parse(decodeURIComponent(feedbackContainer.dataset.feedback));

    // Disable buttons while submitting
    const buttons = feedbackContainer.querySelectorAll('.feedback-btn-mini');
    buttons.forEach(btn => btn.disabled = true);

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

        // Highlight the selected button
        buttons.forEach(btn => btn.classList.remove('selected'));
        button.classList.add('selected');

    } catch (error) {
        console.error('Feedback error:', error);
        // Re-enable buttons on error
        buttons.forEach(btn => btn.disabled = false);
    }
}

// Make submitFeedback available globally
window.submitFeedback = submitFeedback;

// ========================================
// AUTHENTICATION
// ========================================

function showAuthModal(mode) {
    authMode = mode;
    authModal.classList.remove('hidden');
    authError.classList.add('hidden');
    authForm.reset();

    if (mode === 'signup') {
        authModalTitle.textContent = 'Sign Up';
        authSubmitBtn.textContent = 'Create Account';
        nameGroup.classList.remove('hidden');
        authSwitchText.textContent = 'Already have an account?';
        authSwitchBtn.textContent = 'Log In';
    } else {
        authModalTitle.textContent = 'Log In';
        authSubmitBtn.textContent = 'Log In';
        nameGroup.classList.add('hidden');
        authSwitchText.textContent = "Don't have an account?";
        authSwitchBtn.textContent = 'Sign Up';
    }
}

function hideAuthModal() {
    authModal.classList.add('hidden');
}

function toggleAuthMode() {
    showAuthModal(authMode === 'login' ? 'signup' : 'login');
}

async function handleAuthSubmit(event) {
    event.preventDefault();

    authError.classList.add('hidden');
    authSubmitBtn.disabled = true;
    authSubmitBtn.textContent = authMode === 'login' ? 'Logging in...' : 'Creating account...';

    try {
        const endpoint = authMode === 'login' ? '/auth/login' : '/auth/signup';
        const body = {
            email: authEmail.value,
            password: authPassword.value,
        };

        if (authMode === 'signup') {
            body.name = authName.value || null;
        }

        const response = await fetch(`${API_BASE}${endpoint}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || 'Authentication failed');
        }

        // Success - update UI
        currentUser = data.user;
        updateAuthUI();
        hideAuthModal();
        loadRecentSearches();

    } catch (error) {
        authError.textContent = error.message;
        authError.classList.remove('hidden');
    } finally {
        authSubmitBtn.disabled = false;
        authSubmitBtn.textContent = authMode === 'login' ? 'Log In' : 'Create Account';
    }
}

async function logout() {
    try {
        await fetch(`${API_BASE}/auth/logout`, { method: 'POST' });
    } catch (error) {
        console.error('Logout error:', error);
    }

    currentUser = null;
    updateAuthUI();
    recentSearchesSection.classList.add('hidden');
}

function updateAuthUI() {
    if (currentUser) {
        headerActions.classList.add('hidden');
        userMenu.classList.remove('hidden');
        userName.textContent = currentUser.name || currentUser.email;
    } else {
        headerActions.classList.remove('hidden');
        userMenu.classList.add('hidden');
    }
}

async function checkAuthStatus() {
    try {
        const response = await fetch(`${API_BASE}/auth/me`);
        const data = await response.json();

        if (data.user) {
            currentUser = data.user;
            updateAuthUI();
            loadRecentSearches();
        }
    } catch (error) {
        console.error('Auth check error:', error);
    }
}

// Make auth functions global
window.showAuthModal = showAuthModal;
window.hideAuthModal = hideAuthModal;
window.toggleAuthMode = toggleAuthMode;
window.handleAuthSubmit = handleAuthSubmit;
window.logout = logout;

// ========================================
// RECENT SEARCHES
// ========================================

async function loadRecentSearches() {
    if (!currentUser) return;

    try {
        const response = await fetch(`${API_BASE}/auth/history`);
        const data = await response.json();

        if (data.history && data.history.length > 0) {
            recentSearchesSection.classList.remove('hidden');
            recentSearchesList.innerHTML = data.history.map(item => {
                const icon = item.query_type === 'image' ? '📷' : item.query_type === 'url' ? '🔗' : '🔍';
                return `
                    <div class="recent-search-item" onclick="executeRecentSearch('${escapeHtml(item.query)}', '${item.query_type}')">
                        <span class="recent-search-icon">${icon}</span>
                        <span>${escapeHtml(item.query.substring(0, 40))}${item.query.length > 40 ? '...' : ''}</span>
                    </div>
                `;
            }).join('');
        } else {
            recentSearchesSection.classList.add('hidden');
        }
    } catch (error) {
        console.error('Failed to load recent searches:', error);
    }
}

function executeRecentSearch(query, queryType) {
    if (queryType === 'image') {
        // Can't re-execute image searches from history
        return;
    }

    searchInput.value = query;
    performSearch(query, queryType);
}

async function clearSearchHistory() {
    try {
        await fetch(`${API_BASE}/auth/history`, { method: 'DELETE' });
        recentSearchesSection.classList.add('hidden');
    } catch (error) {
        console.error('Failed to clear history:', error);
    }
}

window.executeRecentSearch = executeRecentSearch;
window.clearSearchHistory = clearSearchHistory;

// ========================================
// POPULAR SEARCHES
// ========================================

async function loadPopularSearches() {
    const container = document.getElementById('popular-searches');
    if (!container) return;

    try {
        const response = await fetch(`${API_BASE}/search/popular?days=7&limit=5`);
        if (!response.ok) throw new Error('Failed to load popular searches');

        const data = await response.json();
        const searches = data.searches || [];

        if (searches.length > 0) {
            container.innerHTML = searches.map((item, index) => `
                <div class="popular-item" onclick="searchPopular('${escapeHtml(item.query)}')">
                    <span class="popular-rank">${index + 1}</span>
                    <div class="popular-content">
                        <span class="popular-query">${escapeHtml(item.query)}</span>
                        ${item.search_count > 0 ? `<span class="popular-count">${item.search_count} searches</span>` : ''}
                    </div>
                    <span class="popular-arrow">→</span>
                </div>
            `).join('');
        } else {
            container.innerHTML = `
                <div class="popular-empty">
                    <span class="popular-empty-icon">🔍</span>
                    <p>No searches yet. Be the first to search!</p>
                </div>
            `;
        }
    } catch (error) {
        console.error('Failed to load popular searches:', error);
        container.innerHTML = `
            <div class="popular-empty">
                <span class="popular-empty-icon">✨</span>
                <p>Start searching to see trending products!</p>
            </div>
        `;
    }
}

function searchPopular(query) {
    searchInput.value = query;
    performSearch(query, 'text');
}

window.searchPopular = searchPopular;

// ========================================
// UTILITY FUNCTIONS
// ========================================

function formatPrice(price, currency = 'USD') {
    const symbols = {
        'USD': '$',
        'EUR': '€',
        'GBP': '£',
        'JPY': '¥',
        'SEK': 'kr ',
    };
    const symbol = symbols[currency] || currency + ' ';
    return `${symbol}${price.toFixed(2)}`;
}

function escapeHtml(text) {
    if (!text) return '';
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
        loadingTime.textContent = `${elapsed}s`;
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
        { el: stepSearch, status: 'Searching databases...' },
        { el: stepRank, status: 'Ranking results...' }
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

    // Auto-advance steps
    if (step < 3) {
        setTimeout(() => updateLoadingStep(step + 1), 1200);
    }
}

// ========================================
// SEARCH MODE (Collapsed View)
// ========================================

function enterSearchMode() {
    if (isSearchMode) return;
    isSearchMode = true;

    document.body.classList.add('search-mode');

    // Hide homepage sections with animation
    const homeSections = [
        document.querySelector('.hero-section'),
        document.querySelector('.stats-section'),
        document.querySelector('.how-it-works-section'),
        document.getElementById('popular-section'),
        document.getElementById('recent-searches-section'),
    ];

    homeSections.forEach(section => {
        if (section) {
            section.classList.add('hidden-in-search');
        }
    });

    // Collapse search module
    const searchModule = document.getElementById('search-module');
    if (searchModule) {
        searchModule.classList.add('collapsed');
    }
}

function exitSearchMode() {
    if (!isSearchMode) return;
    isSearchMode = false;

    document.body.classList.remove('search-mode');

    // Show homepage sections
    const homeSections = [
        document.querySelector('.hero-section'),
        document.querySelector('.stats-section'),
        document.querySelector('.how-it-works-section'),
        document.getElementById('popular-section'),
    ];

    homeSections.forEach(section => {
        if (section) {
            section.classList.remove('hidden-in-search');
        }
    });

    // Show recent searches only if logged in
    if (currentUser) {
        const recentSection = document.getElementById('recent-searches-section');
        if (recentSection) {
            recentSection.classList.remove('hidden-in-search');
            loadRecentSearches();
        }
    }

    // Expand search module
    const searchModule = document.getElementById('search-module');
    if (searchModule) {
        searchModule.classList.remove('collapsed');
    }

    // Hide results
    hideResults();
    hideError();

    // Clear search input
    searchInput.value = '';
    clearSelectedImage();

    // Reload popular searches (might have changed)
    loadPopularSearches();

    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

function goHome() {
    exitSearchMode();
}

window.goHome = goHome;

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

window.toggleContributeForm = toggleContributeForm;

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

window.submitProduct = submitProduct;

// ========================================
// KEYBOARD SHORTCUTS
// ========================================

document.addEventListener('keydown', (e) => {
    // Press '/' to focus search
    if (e.key === '/' && document.activeElement.tagName !== 'INPUT') {
        e.preventDefault();
        searchInput.focus();
    }

    // Press 'Escape' to clear image, close modal, or go home
    if (e.key === 'Escape') {
        if (!authModal.classList.contains('hidden')) {
            hideAuthModal();
        } else if (selectedImageFile) {
            clearSelectedImage();
        } else if (isSearchMode) {
            goHome();
        }
    }
});

// ========================================
// PASTE HANDLER
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
                    imageDropZone.classList.add('active');
                    setSelectedImage(file);
                }
                return;
            }
        }
    }
});

// ========================================
// MODAL CLOSE ON OUTSIDE CLICK
// ========================================

authModal.addEventListener('click', (e) => {
    if (e.target === authModal) {
        hideAuthModal();
    }
});

// ========================================
// INITIALIZE
// ========================================

document.addEventListener('DOMContentLoaded', () => {
    console.log('Retail Right initialized');
    searchInput.focus();

    // Check if user is already logged in
    checkAuthStatus();

    // Load popular searches
    loadPopularSearches();
});
