async (page) => {
  page.setDefaultTimeout(15000);
  page.setDefaultNavigationTimeout(15000);
  const check = (value, message) => { if (!value) throw new Error(message); };
  const ready = () => page.waitForFunction(() => {
    const chat = document.querySelector('.chat-layout');
    return chat?.getAttribute('aria-busy') === 'false' && chat.getAttribute('data-route') === location.pathname;
  });
  let eagerCreates = 0;
  const requests = request => {
    if (request.method() === 'POST' && new URL(request.url()).pathname === '/api/chats') eagerCreates++;
  };
  page.on('request', requests);
  try {
    await page.setViewportSize({width:1440,height:960});
    await page.goto('http://127.0.0.1:7867/');
    await page.waitForURL('**/chats/new');
    await ready();
    const originalCount = await page.locator('.conversation-select').count();
    for (let index = 0; index < 3; index++) {
      await page.getByRole('button', {name:'Create', exact:true}).click();
      await page.getByRole('button', {name:'New chat', exact:true}).click();
      check(new URL(page.url()).pathname === '/chats/new', 'Blank Create changed the draft URL');
      check(await page.locator('.conversation-select').count() === originalCount, 'Blank Create added sidebar entries');
    }
    const firstPrompt = 'Lazy first prompt ' + Date.now();
    await page.getByRole('textbox', {name:'Image prompt', exact:true}).fill(firstPrompt);
    await page.locator('input[type=file]').first().setInputFiles('results/qa/supplied-preview.png');
    await page.getByRole('button', {name:'Add reference (1/4)', exact:true}).waitFor();
    await page.getByRole('button', {name:'Create', exact:true}).click();
    check(await page.getByRole('textbox', {name:'Image prompt', exact:true}).inputValue() === firstPrompt, 'Create discarded unsent text');
    check(await page.locator('.attachment').count() === 1, 'Create discarded draft attachments');
    check(eagerCreates === 0, 'The UI persisted an empty chat');
    const response = page.waitForResponse(response => new URL(response.url()).pathname === '/api/jobs' && response.request().method() === 'POST');
    await page.getByRole('button', {name:'Generate image', exact:true}).click();
    const job = await (await response).json();
    check(job.payload?.new_chat === true && job.payload?.chat_id, 'First prompt did not receive its own saved conversation');
    await page.waitForURL('**/chats/' + job.payload.chat_id);
    await page.getByAltText('Generated image', {exact:true}).waitFor();
    check(await page.locator('.conversation-select').count() === originalCount + 1, 'First prompt did not create exactly one sidebar entry');
    check(await page.getByAltText('Generated image', {exact:true}).count() === 1, 'First prompt generated more than one output');
    await page.getByRole('button', {name:'Create', exact:true}).click();
    await page.waitForURL('**/chats/new');
    await ready();
    check(await page.getByRole('textbox', {name:'Image prompt', exact:true}).inputValue() === '', 'The next draft was not blank');
    check(await page.locator('.conversation-select').count() === originalCount + 1, 'Opening another draft persisted it');
    await page.reload();
    await ready();
    check(new URL(page.url()).pathname === '/chats/new', 'Blank reload opened a saved chat');
    check(await page.locator('.conversation-select').count() === originalCount + 1, 'Reload created an empty conversation');
    await page.getByRole('button', {name:'Avatar Studio', exact:true}).click();
    await page.getByRole('button', {name:'Open avatar Enhancement QA', exact:true}).click();
    await ready();
    await page.getByRole('button', {name:'New avatar chat', exact:true}).click();
    await page.waitForURL(/\/avatars\/[^/]+\/console\/new$/);
    await ready();
    const avatarCount = await page.locator('.conversation-select').count();
    await page.getByRole('button', {name:'New avatar chat', exact:true}).click();
    check(await page.locator('.conversation-select').count() === avatarCount, 'Avatar draft created an empty entry');
    await page.getByRole('textbox', {name:'Image prompt', exact:true}).fill('Lazy avatar prompt ' + Date.now());
    await page.getByRole('button', {name:'Generate image', exact:true}).click();
    await page.waitForURL(url => /\/avatars\/[^/]+\/console\/[^/]+$/.test(url.pathname) && !url.pathname.endsWith('/new'));
    await page.getByAltText('Generated image', {exact:true}).waitFor();
    check(await page.locator('.conversation-select').count() === avatarCount + 1, 'Avatar first prompt did not create one entry');
    check(eagerCreates === 0, 'A draft used the eager chat creation endpoint');
    await page.getByRole('button', {name:'Create', exact:true}).click();
    await page.waitForURL('**/chats/new');
    await ready();
    await page.screenshot({path:'output/playwright/lazy-chat-draft-desktop.png'});
    await page.setViewportSize({width:390,height:844});
    check(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), 'Draft mobile overflow');
    await page.screenshot({path:'output/playwright/lazy-chat-draft-mobile.png'});
    await page.setViewportSize({width:1440,height:960});
    return {passed:true, checks:['repeated Create/New chat is a no-op','zero eager chat POSTs','draft text and attachments preserved','one saved chat per first prompt','one generation','blank refresh','avatar draft parity','mobile layout']};
  } finally {
    page.off('request', requests);
  }
}
