function [Xlag] = mlag2(X,p)
%MLAG2  Build a matrix of p lags of X.
%   Xlag = mlag2(X,p) returns a Traw-by-(N*p) matrix whose columns are the
%   1st..p-th lags of the N series in X; the first p rows are zero.
[Traw,N]=size(X);
Xlag=zeros(Traw,N*p);
for ii=1:p
    Xlag(p+1:Traw,(N*(ii-1)+1):N*ii)=X(p+1-ii:Traw-ii,1:N);
end