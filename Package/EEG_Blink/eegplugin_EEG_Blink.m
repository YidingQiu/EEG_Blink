function eegplugin_EEG_Blink(fig,~,~)

toolmenu = findobj(fig, 'tag', 'tools');
uimenu( toolmenu, 'label', 'EEG_Blink','callback', '[EEG LASTCOM]=pop_EEG_Blink(EEG)');