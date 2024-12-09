function [db, mag, pha, grd, w] = freqz_m(b, a, n)
    if nargin < 3
        n = 512; 
    end
    [h, w] = freqz(b, a, n);
    mag = abs(h);
    db = 20*log10((mag+eps)/max(mag));
    pha = -angle(h); 
    grd = grpdelay(b, a, n); 
end
