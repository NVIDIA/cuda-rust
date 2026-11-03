/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Client-side navigation for the cuda-oxide book.
 *
 * Intercepts left-sidebar TOC link clicks and swaps only the main content
 * and right sidebar in place, leaving the left sidebar completely untouched.
 * The TOC never scrolls, repaints, or flashes.
 */
(function () {
    'use strict';

    var SEL = {
        main:        '.bd-article-container',
        rightBar:    '.bd-sidebar-secondary',
        leftTocTree: '.bd-sidebar-primary .sidebar-tree',
        leftBar:     '.bd-sidebar-primary',
    };

    var pending = null;   // track in-flight fetch so rapid clicks don't pile up
    var currentPath = location.pathname + location.search;

    // Resolve the home URL from the navbar brand on first use (not at parse
    // time, because this script loads in <head> before the navbar exists).
    // We capture it once so that later pushState URL changes don't cause the
    // brand's relative href to resolve against the wrong path depth.
    var homeHref = null;
    function getHomeHref() {
        if (homeHref) return homeHref;
        var brand = document.querySelector('.navbar-brand');
        if (brand) homeHref = brand.href;
        return homeHref;
    }

    function scrollTargetForHash(hash) {
        if (!hash || hash === '#') return null;

        var id = decodeURIComponent(hash.slice(1));
        if (!id) return null;

        var target = document.getElementById(id);
        if (!target) return null;

        // Section IDs are attached to the wrapper element, but the heading inside
        // carries the scroll offset via `scroll-margin-top`.
        if (target.tagName === 'SECTION' && target.firstElementChild) {
            return target.firstElementChild;
        }

        return target;
    }

    function scrollForUrl(url) {
        var target = scrollTargetForHash(url.hash);

        if (target) {
            target.scrollIntoView({ behavior: 'smooth' });
            return;
        }

        window.scrollTo({ top: 0, behavior: 'smooth' });
    }

    function jumpToHash(url, pushState) {
        if (pending) {
            pending.abort();
            pending = null;
        }

        if (pushState !== false) {
            var method = url.hash === location.hash ? 'replaceState' : 'pushState';
            history[method]({ hashOnly: true }, document.title, url.href);
        }

        scrollForUrl(url);
    }

    function isPageUrl(url) {
        return url.pathname.endsWith('/') || url.pathname.endsWith('.html');
    }

    function swap(doc) {
        // --- Main content ---
        var newMain = doc.querySelector(SEL.main);
        var curMain = document.querySelector(SEL.main);
        if (newMain && curMain) curMain.replaceWith(newMain);

        // --- Right sidebar (on-this-page TOC) ---
        var newRight = doc.querySelector(SEL.rightBar);
        var curRight = document.querySelector(SEL.rightBar);
        if (newRight && curRight) curRight.replaceWith(newRight);

        // --- Left sidebar tree: swap ONLY the tree to update current-item
        //     highlighting, while keeping the sidebar scroll position. ---
        var leftBar  = document.querySelector(SEL.leftBar);
        var savedTop = leftBar ? leftBar.scrollTop : 0;

        var newTree = doc.querySelector(SEL.leftTocTree);
        var curTree = document.querySelector(SEL.leftTocTree);
        if (newTree && curTree) curTree.replaceWith(newTree);

        if (leftBar) leftBar.scrollTop = savedTop;
    }

    function navigate(url, pushState) {
        var nextUrl = url instanceof URL ? url : new URL(url, location.href);

        if (pending) pending.abort();

        var ctrl = new AbortController();
        pending = ctrl;

        fetch(nextUrl.href, { signal: ctrl.signal })
            .then(function (r) { return r.text(); })
            .then(function (html) {
                pending = null;
                var doc = new DOMParser().parseFromString(html, 'text/html');
                swap(doc);
                currentPath = nextUrl.pathname + nextUrl.search;
                document.title = doc.title;
                if (pushState !== false) {
                    history.pushState({ url: nextUrl.href, spaPage: true }, doc.title, nextUrl.href);
                }
                requestAnimationFrame(function () {
                    scrollForUrl(nextUrl);
                    document.dispatchEvent(new Event('spa:navigate'));
                });
            })
            .catch(function (err) {
                if (err.name === 'AbortError') return;
                // Network error or parse failure — fall back to normal navigation.
                location.href = nextUrl.href;
            });
    }

    // Same-page hash links need explicit handling after SPA swaps so they still
    // land on the correct heading instead of snapping to the top.
    document.addEventListener('click', function (e) {
        var a = e.target.closest('a[href]');
        if (!a || !a.href) return;
        if (e.defaultPrevented || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
        if (a.target && a.target !== '_self') return;
        if (a.hasAttribute('download')) return;

        var url = new URL(a.href, location.href);

        // Let external links, mailto, etc. through.
        if (url.origin !== location.origin) return;

        if (url.hash && url.pathname === location.pathname && url.search === location.search) {
            e.preventDefault();
            jumpToHash(url, true);
            return;
        }

        // Navbar brand (logo / title): SPA-navigate to the saved home URL.
        // After pushState the brand's relative href resolves against the new
        // path depth, so we must use the absolute URL captured at startup.
        var home = getHomeHref();
        if (home && a.closest('.navbar-brand')) {
            e.preventDefault();
            navigate(new URL(home), true);
            return;
        }

        // Intercept internal book page links from the article, sidebars, and
        // navbar so cross-page section references can smooth-scroll after the
        // new page content is swapped in.
        if (!isPageUrl(url)) return;

        e.preventDefault();
        navigate(url, true);
    });

    // Handle browser back / forward.
    window.addEventListener('popstate', function () {
        var url = new URL(location.href);

        if (url.pathname + url.search === currentPath) {
            requestAnimationFrame(function () {
                scrollForUrl(url);
            });
            return;
        }

        navigate(url, false);
    });
})();
