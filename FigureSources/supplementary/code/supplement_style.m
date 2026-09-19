function s = supplement_style()
%SUPPLEMENT_STYLE Shared visual grammar for MLHealth Supplementary Figures.
%   The returned struct centralizes font sizes, line widths, and colors so
%   Figures S1-S11 are regenerated with one consistent style.

s.fontName = 'Helvetica';
s.tickFontSize = 10.5;
s.labelFontSize = 12;
s.legendFontSize = 9.5;
s.panelFontSize = 15;
s.annotationFontSize = 9.5;
s.axisLineWidth = 1.1;
s.mainLineWidth = 2.0;
s.secondaryLineWidth = 1.4;
s.markerSize = 38;
s.largeMarkerSize = 58;
s.gridColor = [0.88 0.88 0.88];
s.gridAlpha = 0.35;
s.blue = [0.00 0.447 0.741];
s.orange = [0.850 0.325 0.098];
s.green = [0.466 0.674 0.188];
s.purple = [0.494 0.184 0.556];
s.red = [0.635 0.078 0.184];
s.cyan = [0.301 0.745 0.933];
s.gray = [0.48 0.48 0.48];
s.lightGray = [0.78 0.78 0.78];
s.veryLightGray = [0.90 0.90 0.90];
s.black = [0.10 0.10 0.10];
s.trainColor = s.blue;
s.validationColor = s.orange;
end
