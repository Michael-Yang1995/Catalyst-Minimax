function [x] = proj_ball(y, a)
% y is the vector, a is diameter 
     m = length(y); 
    
     x = repmat(1/m,m,1)+ (y-1/m)/max(1,norm(y-1/m,2)/a);
    
     
     %x = reshape(x, y,1);
return
