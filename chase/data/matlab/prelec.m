function w = prelec(p,gamma,delta)

r = cumsum(p,2);
wr = exp(-delta.*(-log(r)).^gamma ); 
wr(r==0)= 0;
wr(r==1) =1; 
w(:,2) = wr(:,2)-wr(:,1);
w(:,1) = wr(:,1);