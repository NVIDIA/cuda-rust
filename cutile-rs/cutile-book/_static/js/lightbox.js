// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Simple lightbox for SVG/images in documentation
 * Click on any image to view it enlarged in a modal
 */
document.addEventListener('DOMContentLoaded', function() {
    // Create the lightbox modal elements
    const overlay = document.createElement('div');
    overlay.id = 'lightbox-overlay';
    overlay.innerHTML = `
        <div class="lightbox-content">
            <button class="lightbox-close" aria-label="Close">&times;</button>
            <img class="lightbox-image" src="" alt="Enlarged view">
        </div>
    `;
    document.body.appendChild(overlay);

    const lightboxImage = overlay.querySelector('.lightbox-image');
    const closeBtn = overlay.querySelector('.lightbox-close');

    // Add click handler to all images in the content area
    function initLightbox() {
        const images = document.querySelectorAll('.bd-article-container img, .bd-content img, article img, .figure img');
        
        images.forEach(img => {
            // Skip if already initialized or is an icon
            if (img.dataset.lightboxInit || img.classList.contains('no-lightbox')) {
                return;
            }
            
            // Wait for image to load to check dimensions
            const setupImage = () => {
                if (img.naturalWidth < 150 && img.width < 150) return;
                
                img.style.cursor = 'zoom-in';
                img.title = 'Click to enlarge';
                img.dataset.lightboxInit = 'true';
                
                img.addEventListener('click', function(e) {
                    e.preventDefault();
                    e.stopPropagation();
                    
                    // Get the actual image source
                    let imgSrc = this.src || this.currentSrc;
                    
                    // For SVGs that might be using object/embed, try to get the data attribute
                    if (!imgSrc && this.dataset && this.dataset.src) {
                        imgSrc = this.dataset.src;
                    }
                    
                    if (imgSrc) {
                        lightboxImage.src = imgSrc;
                        lightboxImage.alt = this.alt || 'Enlarged image';
                        overlay.classList.add('active');
                        document.body.style.overflow = 'hidden';
                    }
                });
            };
            
            if (img.complete && img.naturalWidth > 0) {
                setupImage();
            } else {
                img.addEventListener('load', setupImage);
                // Fallback for SVGs that might not fire load event
                setTimeout(setupImage, 500);
            }
        });
    }

    // Close lightbox handlers
    function closeLightbox() {
        overlay.classList.remove('active');
        document.body.style.overflow = '';
        lightboxImage.src = '';
    }

    closeBtn.addEventListener('click', closeLightbox);
    
    overlay.addEventListener('click', function(e) {
        if (e.target === overlay) {
            closeLightbox();
        }
    });

    // Close on Escape key
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape' && overlay.classList.contains('active')) {
            closeLightbox();
        }
    });

    // Initialize after a short delay to ensure images are loaded
    setTimeout(initLightbox, 300);
    
    // Re-initialize when page content might change
    window.addEventListener('load', function() {
        setTimeout(initLightbox, 100);
    });
});
