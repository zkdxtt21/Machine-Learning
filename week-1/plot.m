t = [0:0.01:0.98];
y1 = sin(2*pi*4 *t);
plot(t,y1);
hold on;
y2 = cos(2*pi*4*t);
plot(t,y2);
xlabel('time')
ylabel('value')
legend('sin', 'cos')
title('my plot')
print -dpng 'myPlot.png'
close
figure(1); plot(t,y1);
figure(2); plot(t,y2);
subplot(1,2,1);plot(t,y1);
subplot(1,2,2); plot(t,y2);
axis([0.5 1 -1 1])
clf;
A = magic(5)
imagesc(A)
imagesc(A), colorbar
imagesc(A), colorbar, colormap gray