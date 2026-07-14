() => {
    document.documentElement.setAttribute('lang', 'en');

    const forceDarkMode = () => {
        if (!document.documentElement.classList.contains('dark')) {
            document.documentElement.classList.add('dark');
            document.documentElement.style.colorScheme = 'dark';
        }
    };
    forceDarkMode();

    const darkThemeObserver = new MutationObserver(forceDarkMode);
    darkThemeObserver.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });

    function addAriaLabels() {
        const buttons = document.querySelectorAll('button');
        buttons.forEach(btn => {
            const text = btn.innerText || '';
            if (text.includes('▶️ Start')) {
                btn.setAttribute('aria-label', 'Start Docker inference server');
            } else if (text.includes('⏹️ Stop')) {
                btn.setAttribute('aria-label', 'Stop Docker inference server');
            } else if (text.includes('🔄 Recreate & Run')) {
                btn.setAttribute('aria-label', 'Recreate and run Docker inference container');
            } else if (text.includes('🚀 Start Batch Processing')) {
                btn.setAttribute('aria-label', 'Start batch processing of uploaded PDF files');
            } else if (text.includes('🛑 Stop Process')) {
                btn.setAttribute('aria-label', 'Stop current batch processing pipeline');
            } else if (text.includes('⬅️ Prev Page')) {
                btn.setAttribute('aria-label', 'Go to previous page');
            } else if (text.includes('Next Page ➡️')) {
                btn.setAttribute('aria-label', 'Go to next page');
            } else if (text.includes('📋 Copy')) {
                btn.setAttribute('aria-label', 'Copy raw markdown text to clipboard');
            } else if (text.includes('💾 Save Configuration')) {
                btn.setAttribute('aria-label', 'Save pipeline configuration settings');
            } else if (text.includes('🧹 Clean & Reset')) {
                btn.setAttribute('aria-label', 'Perform cache and workspace cleanup');
            }
        });

        const settingsBtn = document.querySelector('button.settings');
        if (settingsBtn) {
            settingsBtn.setAttribute('aria-label', 'Settings drawer toggle');
            const settingsImg = settingsBtn.querySelector('img');
            if (settingsImg && !settingsImg.getAttribute('alt')) {
                settingsImg.setAttribute('alt', 'Settings icon');
            }
        }

        const svgs = document.querySelectorAll('svg');
        svgs.forEach(svg => {
            if (!svg.getAttribute('aria-hidden') && !svg.querySelector('title')) {
                svg.setAttribute('aria-hidden', 'true');
            }
        });
    }

    addAriaLabels();

    const accessibilityObserver = new MutationObserver(addAriaLabels);
    accessibilityObserver.observe(document.body, { childList: true, subtree: true });

    document.addEventListener('keydown', function(e) {
        if (e.ctrlKey && !e.shiftKey && e.key === 'Enter') {
            const askBtn = document.querySelector('button');
            const allBtns = document.querySelectorAll('button');
            for (const btn of allBtns) {
                if (btn.innerText && btn.innerText.includes('🚀 Ask')) {
                    e.preventDefault();
                    btn.click();
                    break;
                }
            }
        }
        if (e.ctrlKey && e.shiftKey && e.key === 'N') {
            const allBtns = document.querySelectorAll('button');
            for (const btn of allBtns) {
                if (btn.innerText && btn.innerText.includes('Clear Chat')) {
                    e.preventDefault();
                    btn.click();
                    break;
                }
            }
        }
        if (e.ctrlKey && e.shiftKey && e.key === 'C') {
            const botMsgs = document.querySelectorAll('.analysis-chatbot .bot, .analysis-chatbot .message.bot');
            if (botMsgs.length > 0) {
                const lastBot = botMsgs[botMsgs.length - 1];
                const text = lastBot.innerText || lastBot.textContent || '';
                if (text) {
                    e.preventDefault();
                    navigator.clipboard.writeText(text).catch(() => {});
                }
            }
        }
    });

    let activeScrollSource = null;
    let scrollTimeout = null;

    function findScrollableElement(el) {
        if (!el) return null;
        if (el.scrollHeight > el.clientHeight && (window.getComputedStyle(el).overflowY === 'auto' || window.getComputedStyle(el).overflowY === 'scroll')) {
            return el;
        }
        for (let i = 0; i < el.children.length; i++) {
            const found = findScrollableElement(el.children[i]);
            if (found) return found;
        }
        return null;
    }

    window.addEventListener('scroll', function(event) {
        const syncCheckboxContainer = document.getElementById('sync-scroll-checkbox');
        const syncCheckbox = syncCheckboxContainer ? syncCheckboxContainer.querySelector('input[type="checkbox"]') : null;
        if (!syncCheckbox || !syncCheckbox.checked) return;

        const source = event.target;
        if (!source) return;

        const pdf = document.getElementById('pdf-scroll-container');
        const raw = document.getElementById('raw-scroll-container');
        const previewContainer = document.getElementById('preview-scroll-container');
        const preview = findScrollableElement(previewContainer) || previewContainer;

        let sourceWindow = null;
        if (pdf && (pdf === source || pdf.contains(source))) {
            sourceWindow = pdf;
        } else if (raw && (raw === source || raw.contains(source))) {
            sourceWindow = raw;
        } else if (preview && (preview === source || preview.contains(source))) {
            sourceWindow = preview;
        }

        if (!sourceWindow) return;

        if (activeScrollSource && activeScrollSource !== sourceWindow) return;
        activeScrollSource = sourceWindow;

        const maxSourceScroll = sourceWindow.scrollHeight - sourceWindow.clientHeight;
        if (maxSourceScroll <= 0) return;
        const percentage = sourceWindow.scrollTop / maxSourceScroll;

        const targets = [pdf, raw, preview].filter(el => el && el !== sourceWindow);

        targets.forEach(target => {
            const maxTargetScroll = target.scrollHeight - target.clientHeight;
            target.scrollTop = percentage * maxTargetScroll;
        });

        clearTimeout(scrollTimeout);
        scrollTimeout = setTimeout(() => {
            activeScrollSource = null;
        }, 100);
    }, true);
}
