function version = eegplugin_EEG_Blink(fig,~,~)
version = "0.1 building...";


%toolmenu = findobj(fig, 'tag', 'tools');
toolmenu = findobj(fig, 'tag', 'tools');
uimenu( toolmenu, 'label', 'EEG_Blink','callback', '[EEG LASTCOM]=pop_EEG_Blink(EEG)');