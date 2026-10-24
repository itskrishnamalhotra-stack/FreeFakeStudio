async (page) => {
  const errors = [];
  const origin = 'http://127.0.0.1:7867';
  const request = {
    get: (url) => page.request.get(origin + url),
    post: (url, options) => page.request.post(origin + url, options),
  };
  page.on('pageerror', error => errors.push(error.message));
  const config = await (await request.get('/api/config')).json();
  if (!config.mock) throw new Error('This browser test requires mock mode');
  const fixture = (await (await request.get('/api/avatars')).json()).find(row => row.name === 'Enhancement QA');
  if (!fixture) throw new Error('Run python prepare_browser_fixtures.py before browser tests');
  const avatar = await (await request.post('/api/avatars', { data: { name: 'Influencer Browser ' + Date.now(), mode: 'Custom' } })).json();
  const base = '/api/avatars/' + avatar.id;
  async function waitJob(id) {
    for (let index = 0; index < 600; index++) {
      const job = await (await request.get('/api/jobs/' + id)).json();
      if (!['queued', 'running'].includes(job.status)) {
        if (job.status !== 'completed') throw new Error(JSON.stringify(job));
        return job;
      }
      await page.waitForTimeout(100);
    }
    throw new Error('Mock job timeout');
  }
  for (const part of ['face', 'body']) {
    const response = await request.post('/api/jobs', { headers: { 'Idempotency-Key': crypto.randomUUID() }, data: { kind: 'confirm', avatar_id: avatar.id, reference_kind: part, image: fixture[part].path } });
    await waitJob((await response.json()).id);
  }
  await page.setViewportSize({ width: 1366, height: 768 });
  await page.goto(origin + '/avatars/' + avatar.id + '/identity');
  await page.getByRole('heading', { name: 'Identity references', exact: true }).waitFor();
  await page.getByLabel('Reference view', { exact: true }).selectOption('profile');
  await page.getByLabel('Additional details', { exact: true }).fill('Natural side portrait');
  await page.getByRole('button', { name: 'Generate reference', exact: true }).click();
  await page.getByRole('button', { name: 'Approve identity', exact: true }).waitFor({ timeout: 30000 });
  await page.getByRole('button', { name: 'Approve identity', exact: true }).click();
  await page.getByRole('button', { name: 'Unapprove', exact: true }).waitFor();
  const pack = await (await request.get(base + '/identity/export')).body();
  if (pack[0] !== 80 || pack[1] !== 75) throw new Error('Pack export is not ZIP');
  await page.screenshot({ path: 'output/playwright/influencer-identity-desktop.png' });
  await page.getByRole('tab', { name: 'Quality', exact: true }).click();
  await page.getByRole('heading', { name: 'Consistency policy', exact: true }).waitFor();
  await page.getByLabel('hair color', { exact: true }).selectOption('required');
  await page.getByRole('button', { name: 'Save policy', exact: true }).click();
  await page.waitForFunction(async ({ base }) => (await (await fetch(base + '/identity')).json()).profile?.policies?.hair_color === 'required', { base });
  await page.getByRole('button', { name: 'Run 12-image test', exact: true }).click();
  await page.waitForFunction(async ({ base }) => (await (await fetch(base + '/identity')).json()).benchmark.completed === 12, { base }, { timeout: 60000 });
  await page.getByLabel('Human assessment 1', { exact: true }).waitFor({ timeout: 30000 });
  await page.getByLabel('Human assessment 1', { exact: true }).selectOption('wrong');
  await page.reload();
  await page.getByLabel('hair color', { exact: true }).waitFor();
  if (await page.getByLabel('hair color', { exact: true }).inputValue() !== 'required') throw new Error('Policy lost after reload');
  await page.getByRole('tab', { name: 'Content', exact: true }).click();
  await page.getByLabel('Niche', { exact: true }).waitFor();
  await page.getByLabel('Niche', { exact: true }).fill('Travel portraits');
  await page.getByLabel('Tone', { exact: true }).fill('Warm and concise');
  await page.getByLabel('Disclosure', { exact: true }).fill('AI-created persona');
  await page.getByRole('button', { name: 'Save persona', exact: true }).click();
  await page.waitForFunction(async ({ base }) => (await (await fetch(base + '/content')).json()).persona.niche === 'Travel portraits', { base });
  const option = await page.getByLabel('Photograph', { exact: true }).locator('option').nth(1).getAttribute('value');
  await page.getByLabel('Photograph', { exact: true }).selectOption(option);
  await page.getByLabel('Caption brief', { exact: true }).fill('A quiet afternoon portrait');
  await page.getByLabel('Caption style', { exact: true }).selectOption('editorial');
  await page.getByRole('button', { name: 'Compose caption', exact: true }).click();
  await page.waitForFunction(() => document.querySelector('textarea[maxlength="2200"]')?.value.includes('AI-created persona'));
  await page.getByRole('button', { name: 'Save draft', exact: true }).click();
  await page.getByRole('button', { name: 'Approve post', exact: true }).waitFor();
  const exportLink = page.getByRole('link', { name: 'Export', exact: true });
  const exported = await request.get(await exportLink.getAttribute('href'));
  if (!exported.ok()) throw new Error('Content export failed');
  page.once('dialog', dialog => dialog.accept());
  await page.getByRole('button', { name: 'Approve post', exact: true }).click();
  await page.waitForFunction(async ({ base }) => (await (await fetch(base + '/content')).json()).posts[0]?.status === 'approved', { base });
  if (await page.getByRole('button', { name: 'Publish now', exact: true }).count()) throw new Error('Publishing must be disabled in mock preview');
  if (!await page.getByRole('checkbox', {name:'Runtime scheduling', exact:true}).isDisabled()) throw new Error('Runtime scheduling must be disabled in mock preview');
  await page.screenshot({ path: 'output/playwright/influencer-content-desktop.png' });
  for (const [width, height] of [[390, 844], [320, 740]]) {
    await page.setViewportSize({ width, height });
    for (const tab of ['Identity', 'Quality', 'Content']) {
      await page.getByRole('tab', { name: tab, exact: true }).click();
      await page.locator('.influencer-workspace[aria-busy="false"]').waitFor();
      const overflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth || [...document.querySelectorAll('.influencer-workspace')].some(el => el.scrollWidth > el.clientWidth));
      if (overflow) throw new Error('Horizontal overflow at ' + width + ' in ' + tab);
      await page.screenshot({ path: 'output/playwright/influencer-' + tab.toLowerCase() + '-' + width + '.png' });
    }
  }
  await page.setViewportSize({ width: 1366, height: 768 });
  await page.goto(origin + '/avatars/' + avatar.id + '/identity');
  if (errors.length) throw new Error(JSON.stringify(errors));
  await page.evaluate(result => { window.__ffsInfluencerTest = result; }, { passed: true, avatar: avatar.id });
  console.log(JSON.stringify({ status: 'passed', avatar: avatar.id, mockOnly: true, tested: ['identity candidate approval', 'pack export', 'policy save/reload', '12-case benchmark', 'human label', 'offline caption', 'post approval/export', 'mobile 390/320'] }));
}
