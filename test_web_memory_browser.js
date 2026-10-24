async (page) => {
  const origin = 'http://127.0.0.1:7867';
  const initial = await (await page.request.get(origin + '/api/settings')).json();
  const config = await (await page.request.get(origin + '/api/config')).json();
  if (!config.mock) throw new Error('Only run against the local mock preview');
  const set = value => page.request.patch(origin + '/api/settings', {data:{gpu_memory_management:value}});
  try {
    await set(false);
    await page.setViewportSize({width:1366,height:768});
    await page.goto(origin + '/chats/new');
    await page.getByRole('button', {name:'Settings',exact:true}).click();
    let control = page.getByRole('switch', {name:'GPU memory management',exact:true});
    if (await control.isChecked()) throw new Error('Expected Off');
    await control.check();
    await page.waitForFunction(async () => (await (await fetch('/api/settings')).json()).gpu_memory_management === true);
    await page.getByRole('dialog', {name:'Generation settings'}).getByRole('button', {name:'Close',exact:true}).click();
    await page.getByRole('button', {name:'Avatar Studio',exact:true}).click();
    await page.getByRole('button', {name:'Open avatar Enhancement QA',exact:true}).click();
    await page.getByRole('tab', {name:'Quality',exact:true}).click();
    await page.getByRole('heading', {name:'Consistency policy',exact:true}).waitFor();
    if (await page.getByText('Sequential GPU memory mode', {exact:true}).count()) throw new Error('Legacy avatar switch remains');
    await page.reload();
    await page.getByRole('button', {name:'Settings',exact:true}).click();
    control = page.getByRole('switch', {name:'GPU memory management',exact:true});
    if (!await control.isChecked()) throw new Error('Setting lost between main chat/avatar/reload');
    await control.uncheck();
    await page.waitForFunction(async () => (await (await fetch('/api/settings')).json()).gpu_memory_management === false);
    await page.waitForFunction(() => {
      const control = document.querySelector('[role="switch"][aria-label="GPU memory management"]');
      return control && !control.checked && !control.disabled;
    });
    await page.screenshot({path:'output/playwright/app-memory-desktop.png'});
    for (const width of [390,320]) {
      await page.setViewportSize({width,height:844});
      if (await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)) throw new Error('Memory setting overflows mobile');
      const box = await control.boundingBox();
      if (!box || box.width < 14 || box.width > 24) throw new Error('Switch dimensions incorrect');
      if (await control.isChecked()) throw new Error('Stale refresh overwrote the saved Off setting');
      await page.screenshot({path:`output/playwright/app-memory-${width}.png`});
    }
    await page.keyboard.press('Escape');
    return {passed:true,checks:['app-wide switch','off default','main/avatar shared value','reload persistence','legacy control removed','desktop and mobile']};
  } finally {
    await set(initial.gpu_memory_management);
    await page.setViewportSize({width:1366,height:768});
  }
}
