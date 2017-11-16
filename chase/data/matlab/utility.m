function util = utility(x,alpha,beta,lambda)

util = x;
util(x>=0) = x(x>=0).^alpha;
util(x<0) = -lambda.*abs(x(x<0)).^beta;