// Run with playwright-cli run-code --filename test_avatar_console_ui.js
// Start on the Console tab of a saved local mock avatar. No inference is requested.
async (page) => {
    const assert = (condition, message) => { if (!condition) throw new Error(message); };
    const compose = page.locator('#av-console-compose');
    const chat = page.locator('#av-console-chat');
    const results = [];
    const status = page.locator('#av-console-status');
    for (const className of ['ffs-notice', 'ffs-generation-stage']) {
        await status.evaluate((el, name) => {
            const fixture = document.createElement('div');
            fixture.id = 'console-status-fixture';
            fixture.className = name;
            fixture.textContent = 'Layout fixture';
            el.appendChild(fixture);
        }, className);
        assert(await status.isVisible(), 'Status hidden by scoped CSS: ' + className);
        await status.evaluate(() => document.querySelector('#console-status-fixture').remove());
    }
    for (const [width, height] of [[1440, 1000], [1280, 720], [768, 1024], [390, 844], [320, 568]]) {
        await page.setViewportSize({width, height});
        await page.waitForFunction(() => {
            const chat = document.querySelector('#av-console-chat').getBoundingClientRect();
            const dock = document.querySelector('#av-console-compose').getBoundingClientRect();
            return dock.bottom <= innerHeight && chat.bottom <= dock.top - 10;
        });
        const dock = await compose.boundingBox();
        const view = await chat.boundingBox();
        assert(dock.x >= 0 && dock.x + dock.width <= width + 1, 'Dock overflow at ' + width);
        assert(height - dock.y - dock.height <= 24, 'Composer is not at the bottom');
        assert(view.height >= (height >= 720 ? height * 0.35 : 80), 'Conversation too short');
        assert(await compose.evaluate(el => getComputedStyle(el).position === 'fixed'), 'Composer not fixed');
        await compose.getByRole('button', {name: 'Settings', exact: true}).click();
        const settings = page.locator('#av-console-settings [data-testid="accordion-content"]');
        await settings.waitFor({state: 'visible'});
        const panel = await settings.boundingBox();
        assert(panel.y >= 64 && panel.y + panel.height < dock.y, 'Settings obscure header or composer');
        await compose.getByRole('button', {name: /^References/}).click();
        await settings.waitFor({state: 'hidden'});
        const references = page.locator('#av-console-references [data-testid="accordion-content"]');
        await references.waitFor({state: 'visible'});
        const refPanel = await references.boundingBox();
        assert(refPanel.x >= 0 && refPanel.x + refPanel.width <= width + 1, 'Reference panel overflow');
        assert(refPanel.y >= 64 && refPanel.y + refPanel.height < dock.y, 'References obscure header or composer');
        await page.keyboard.press('Escape');
        await references.waitFor({state: 'hidden'});
        await page.screenshot({path: 'output/playwright/console-footer-' + width + '.png'});
        results.push({width, height, conversationHeight: view.height, footerHeight: dock.height});
    }
    await page.setViewportSize({width: 1440, height: 1000});
    await compose.getByRole('textbox', {name: 'Prompt', exact: true}).fill('Keep this draft when changing tabs');
    await page.getByRole('radio', {name: 'Gallery', exact: true}).click();
    await compose.waitFor({state: 'hidden'});
    await page.waitForFunction(() => !document.documentElement.dataset.ffsAvatarConsole);
    await page.getByRole('radio', {name: 'Console', exact: true}).click();
    await compose.waitFor({state: 'visible'});
    assert(await compose.getByRole('textbox', {name: 'Prompt', exact: true}).inputValue() === 'Keep this draft when changing tabs', 'Draft lost');
    await compose.getByRole('textbox', {name: 'Prompt', exact: true}).fill('');
    await chat.evaluate(el => {
        const fixture = document.createElement('div');
        fixture.id = 'console-scroll-fixture';
        fixture.style.height = '2000px';
        el.appendChild(fixture);
    });
    await page.waitForFunction(() => {
        const el = document.querySelector('#av-console-chat');
        return el.scrollHeight - el.scrollTop - el.clientHeight < 72;
    });
    await chat.evaluate(el => { el.scrollTop = 0; });
    await page.waitForFunction(() => document.querySelector('#av-console-chat').scrollTop === 0);
    await chat.evaluate(el => { document.querySelector('#console-scroll-fixture').appendChild(document.createTextNode('New message')); });
    await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
    assert(await chat.evaluate(el => el.scrollTop) === 0, 'Reading history was interrupted');
    await chat.evaluate(el => { document.querySelector('#console-scroll-fixture').remove(); el.scrollTop = el.scrollHeight; });
    return {layouts: results, exclusivePanels: true, escapeDismisses: true, draftPreserved: true, readingPositionPreserved: true};
}
