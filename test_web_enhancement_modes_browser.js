async (page) => {
  const origin = 'http://127.0.0.1:7867';
  const config = await (await page.request.get(origin + '/api/config')).json();
  if (!config.mock) throw new Error('Mock preview required');
  await page.setViewportSize({width:1366,height:900});
  await page.goto(origin + '/chats/new');
  await page.getByRole('textbox', {name:'Image prompt',exact:true}).fill('Enhancement mode regression');
  await page.getByRole('button', {name:'Generate image',exact:true}).click();
  await page.getByAltText('Generated image', {exact:true}).waitFor();
  await page.getByAltText('Generated image', {exact:true}).hover();
  await page.getByRole('button', {name:'Enhance',exact:true}).first().click();
  const dialog = page.getByRole('dialog', {name:'Enhance image',exact:true});
  const profile = dialog.getByRole('combobox', {name:'Prompt profile',exact:true});
  if (await profile.locator('option').count() !== 16) throw new Error('Missing profile choices');
  await profile.selectOption('female_body');
  await dialog.getByText('Selected profile prompt', {exact:true}).click();
  await dialog.getByRole('button', {name:'Use as custom prompt',exact:true}).click();
  if (await dialog.getByRole('checkbox', {name:'Automatic image-aware prompt',exact:true}).isChecked()) throw new Error('Customizing a profile did not enter custom mode');
  const custom = dialog.getByRole('textbox', {name:'Custom enhancement prompt',exact:true});
  if (await custom.inputValue() !== config.enhancement_profiles.find(row => row.id === 'female_body').prompt) throw new Error('Selected profile text not copied exactly');
  await custom.fill('');
  if (!await dialog.getByRole('button', {name:'Enhance image',exact:true}).isDisabled()) throw new Error('Empty custom prompt accepted');
  const prompt = 'Refine only the fabric stitching. Keep identity and composition unchanged.';
  await custom.fill(prompt);
  for (const width of [1366,390,320]) {
    await page.setViewportSize({width,height:900});
    if (await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)) throw new Error('Prompt controls overflow viewport');
    await page.screenshot({path:`output/playwright/enhancement-custom-${width}.png`});
  }
  await page.setViewportSize({width:1366,height:900});
  const response = page.waitForResponse(r => new URL(r.url()).pathname === '/api/jobs' && r.request().method() === 'POST');
  await dialog.getByRole('button', {name:'Enhance image',exact:true}).click();
  const job = await (await response).json();
  if (job.payload.adaptive_prompt || job.payload.enhancement_profile !== 'auto' || job.payload.additional_instructions) throw new Error('Automatic instructions leaked into custom mode');
  if (job.payload.reconstruction_prompt !== prompt) throw new Error('Custom prompt changed before submission');
  let row;
  for (let index=0;index<200;index++) {
    row = await (await page.request.get(origin + '/api/jobs/' + job.id)).json();
    if (!['queued','running'].includes(row.status)) break;
    await page.waitForTimeout(100);
  }
  if (row.status !== 'completed') throw new Error(JSON.stringify(row));
  if (row.checkpoint.enhancement_plan.prompt !== prompt || row.checkpoint.enhancement_plan.automatic) throw new Error('Applied custom prompt differs from user text');
  await page.reload();
  await page.getByAltText('Generated image', {exact:true}).last().hover();
  await page.getByRole('button', {name:'Enhance',exact:true}).last().click();
  const reopened = page.getByRole('dialog', {name:'Enhance image',exact:true});
  if (!await reopened.getByRole('checkbox', {name:'Automatic image-aware prompt',exact:true}).isChecked()) throw new Error('Old mode leaked into another image');
  if (await reopened.getByRole('textbox', {name:'Additional instructions (optional)',exact:true}).inputValue()) throw new Error('Stale instructions leaked into another image');
  await reopened.getByText('Automatic profile: pending source analysis', {exact:true}).waitFor();
  await reopened.getByRole('combobox', {name:'Prompt profile',exact:true}).selectOption('female_body');
  const overrideResponse = page.waitForResponse(r => new URL(r.url()).pathname === '/api/jobs' && r.request().method() === 'POST');
  await reopened.getByRole('button', {name:'Enhance image',exact:true}).click();
  const override = await (await overrideResponse).json();
  for (let index=0;index<200;index++) {
    row = await (await page.request.get(origin + '/api/jobs/' + override.id)).json();
    if (!['queued','running'].includes(row.status)) break;
    await page.waitForTimeout(100);
  }
  if (row.status !== 'completed') throw new Error(JSON.stringify(row));
  if (row.checkpoint.enhancement_plan.profile_id !== 'female_body' || row.checkpoint.enhancement_plan.selection !== 'manual_override') throw new Error('Profile override ignored');
  if (row.checkpoint.enhancement_plan.requested_prompt) throw new Error('Unused generic prompt incorrectly recorded as user instructions');
  await page.reload();
  await page.getByAltText('Generated image', {exact:true}).last().click();
  await page.getByText('Enhancement analysis', {exact:true}).click();
  await page.getByText('Female-Presenting Body / Fashion', {exact:true}).waitFor();
  await page.getByText('Profile selection: manual override', {exact:true}).waitFor();
  await page.getByText('Sampling details', {exact:true}).click();
  if (!await page.getByRole('dialog', {name:'Image',exact:true}).locator('pre').filter({hasText:'Flux2Scheduler'}).count()) throw new Error('Actual sampler details not visible');
  await page.keyboard.press('Escape');
  return {passed:true,checks:['15 profiles','editable profile template','custom verbatim','empty custom validation','no automatic text leakage','fresh image resets','manual override provenance','sampler details','desktop/mobile controls']};
}
