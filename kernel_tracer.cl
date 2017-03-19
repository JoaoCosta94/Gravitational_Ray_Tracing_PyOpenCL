float8 f(float s,  float8 q)
{
    // Equation system for a photon affected by the gravity
    // (metric) of a Black Hole
    float8 v;
    float phi, cos_psi, k, r, y11, y12, y21;

    r = sqrt(q.s1 * q.s1 + q.s2 * q.s2 + q.s3 * q.s3);
    k = sqrt(q.s5 * q.s5 + q.s6 * q.s6 + q.s7 * q.s7);
    cos_psi = -(q.s1 * q.s5 + q.s2 * q.s6 + q.s3 * q.s7)/(r * k);
    phi = -1.0/r;

    y12 = 2.0 * (1.0 + phi);
    y11 = y12 * k * phi * cos_psi/ r;
    y21 = k * k * phi * (1.0 + (3.0 + 4.0 * phi) * cos_psi * cos_psi)/(r * r);


	v.s0 = q.s4;
	v.s1 = y11 * q.s1 + y12 * q.s5;
	v.s2 = y11 * q.s2 + y12 * q.s6;
	v.s3 = y11 * q.s3 + y12 * q.s7;
	
	v.s4 = -q.s0;
	v.s5 = y21 * q.s1 - y11 * q.s5;
	v.s6 = y21 * q.s2 - y11 * q.s6;
	v.s7 = y21 * q.s3 - y11 * q.s7;

    return v;
}





__kernel void RK4Step(float s, float ds, __global float8 *q){
    const int gid = get_global_id(0);
    float8 k, qm,qs;

    // 4th order Runge-Kutta pusher

    //k1
    k = f(s, q[gid]);
    qs = q[gid] + ds * k/6;
    qm = q[gid] +  ds * k/2;

    //k2
    k = f(s+0.5*ds, qm);
    qs +=  ds * k/3;
    qm = q[gid] +  ds * k/2;

    //k3
    k = f(s+0.5*ds,  qm);
    qs +=  ds * k/3;
    qm = q[gid] + ds * k;

    //k4
    k = f(s + ds, qm);
    qs +=  ds * k/6;

    //update photon
    q[gid] = qs;
}