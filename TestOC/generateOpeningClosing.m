function argout = generateOpeningClosing(signalSeqCell, lableSeqCell,categroies)
    openCloseLables = {};
    blinkLocation = {};
    for n = 1:numel(signalSeqCell)
        signalSeq = signalSeqCell{n};
        lableSeq = lableSeqCell{n};
        locations = blinkLocate(signalSeq, lableSeq);
        blinkLocation{n} = locations;
        for blink = locations
            i = blink;
            while i > 0 && lableSeq(i)=='blink'
                lableSeq(i) = categroies{1};
                i = i-1;
            end
            i = blink+1;
            while i < length(lableSeq)+1 && lableSeq(i)=='blink'
                lableSeq(i) = categroies{2};
                i = i+1;
            end

        end
        lableSeq(lableSeq=='blink')='n/a';
        openCloseLables{n} = categorical(lableSeq,categroies);
        lableSeq(lableSeq=='<undefined>')='n/a';
    end
    argout = {openCloseLables, blinkLocation};
end