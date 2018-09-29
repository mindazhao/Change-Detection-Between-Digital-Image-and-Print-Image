a=300111253;
sum1=0;
for i=1:9
    
    sum1=sum1+mod(a,10);
    a=fix(a/10);
end
bitxor(180,sum1)
x65=dec2hex(bitxor(180,sum1));
x66=dec2hex(bitxor(50,sum1));
disp(num2str(x65));
disp(num2str(x66));
