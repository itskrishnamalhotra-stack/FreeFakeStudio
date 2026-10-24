async (page) => {
  const check = (value, message) => { if (!value) throw new Error(message); };
  const loaded = () => page.waitForFunction(() => document.querySelector('#gallery-panel')?.getAttribute('aria-busy') === 'false');
  const choose = async name => {
    await page.getByRole('tablist', {name:'Gallery sections', exact:true}).getByRole('tab', {name:new RegExp('^' + name + ' ')}).click();
    await loaded();
  };
  await page.setViewportSize({width:1440, height:960});
  await page.reload();
  await page.getByRole('button', {name:'Avatar Studio', exact:true}).click();
  await page.getByRole('button', {name:'Open avatar Enhancement QA', exact:true}).click();
  await page.getByRole('tab', {name:'Gallery', exact:true}).click();
  await loaded();
  check(await page.locator('.gallery-tile').count() > 0, 'Generation gallery empty');
  check(await page.locator('.tile-caption').filter({hasText:'enhancement'}).count() === 0, 'Enhancements leaked into generations');
  await page.screenshot({path:'output/playwright/gallery-generations-desktop.png'});
  await choose('Enhancements');
  check(await page.locator('.gallery-tile').count() > 0, 'Enhancement gallery empty');
  check(await page.getByRole('combobox', {name:'Gallery source', exact:true}).count() === 0, 'Redundant enhancement source selector');
  await page.locator('.tile-caption').getByRole('button', {name:'Compare', exact:true}).first().click();
  await page.getByAltText('Original', {exact:true}).waitFor();
  await page.getByAltText('Reconstruction', {exact:true}).waitFor();
  await page.getByRole('combobox', {name:'Comparison area', exact:true}).selectOption('0');
  await page.waitForFunction(() => ['Original', 'Reconstruction'].every(name => {
    const image = document.querySelector(`img[alt="${name}"]`);
    return image?.complete && image.naturalWidth > 0 && image.src.includes('region=');
  }));
  for (const width of [1366, 390, 320]) {
    await page.setViewportSize({width, height:844});
    check(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), 'Detail comparison overflows viewport');
    await page.screenshot({path:`output/playwright/detail-comparison-${width}.png`});
  }
  await page.setViewportSize({width:1440, height:960});
  await page.getByRole('combobox', {name:'Comparison area', exact:true}).selectOption('full');
  await page.screenshot({path:'output/playwright/gallery-enhancement-compare.png'});
  await page.keyboard.press('Escape');
  await page.locator('.gallery-tile').first().hover();
  await page.getByRole('button', {name:'Enhance image', exact:true}).first().click();
  await page.getByRole('textbox', {name:'Reconstruction prompt', exact:true}).waitFor();
  await page.keyboard.press('Escape');
  await choose('Resources');
  check(await page.locator('.resource-tile').count() > 0, 'Resource inventory empty');
  await page.getByRole('combobox', {name:'Resource type', exact:true}).selectOption('identity');
  await loaded();
  check(await page.locator('.resource-tile').count() >= 2, 'Identity anchors missing');
  await page.locator('.resource-image').first().click();
  await page.getByRole('dialog', {name:'Image', exact:true}).waitFor();
  check(await page.getByRole('dialog').getByRole('button', {name:'Enhance', exact:true}).count() === 0, 'Reference viewer offers output-only enhancement');
  await page.keyboard.press('Escape');
  await page.getByRole('combobox', {name:'Resource type', exact:true}).selectOption('all');
  await page.getByRole('combobox', {name:'Resource status', exact:true}).selectOption('rejected');
  await loaded();
  check(await page.getByRole('heading', {name:'No matching resources', exact:true}).count() === 1, 'Rejected filter did not show empty state');
  await page.getByRole('combobox', {name:'Resource status', exact:true}).selectOption('all');
  await loaded();
  await page.screenshot({path:'output/playwright/gallery-resources-desktop.png'});
  await choose('Needs Review');
  check(await page.locator('.gallery-tile').count() > 0, 'Review queue missing unchecked images');
  const before = await page.locator('.gallery-tile').count();
  await page.locator('.gallery-image').first().click();
  await page.getByRole('button', {name:'Approve', exact:true}).click();
  await page.waitForFunction(count => document.querySelectorAll('.gallery-tile').length < count, before);
  await page.keyboard.press('Escape');
  await choose('Resources');
  for (const viewport of [{width:390,height:844}, {width:320,height:740}]) {
    await page.setViewportSize(viewport);
    check(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), 'Gallery overflows mobile viewport');
    await page.screenshot({path:`output/playwright/gallery-resources-${viewport.width}.png`});
  }
  await page.setViewportSize({width:1440, height:960});
  const resourcesTab = page.getByRole('tablist', {name:'Gallery sections', exact:true}).getByRole('tab', {name:/^Resources /});
  await resourcesTab.focus();
  await page.keyboard.press('ArrowRight');
  await loaded();
  check(await page.getByRole('tablist', {name:'Gallery sections', exact:true}).getByRole('tab', {name:/^Needs Review /}).getAttribute('aria-selected') === 'true', 'Gallery keyboard navigation failed');
  await choose('Generations');
  console.log(JSON.stringify({passed:true,checks:['category isolation','counts','enhancement comparison','prompt editor retained','identity resources','resource fullscreen','resource filters','review approval','390px and 320px layouts','keyboard tabs']}));
}
