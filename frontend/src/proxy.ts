import { timingSafeEqual } from "node:crypto";
import { NextResponse, type NextRequest } from "next/server";

function credentialsMatch(provided: string, expected: string) {
  const providedBytes = Buffer.from(provided);
  const expectedBytes = Buffer.from(expected);

  return (
    providedBytes.length === expectedBytes.length &&
    timingSafeEqual(providedBytes, expectedBytes)
  );
}

export function proxy(request: NextRequest) {
  if (process.env.NODE_ENV !== "production") {
    return NextResponse.next();
  }

  const expectedUsername = process.env.SIGNAL_DESK_AUTH_USER;
  const expectedPassword = process.env.SIGNAL_DESK_AUTH_PASSWORD;

  if (!expectedUsername || !expectedPassword) {
    return new Response("App authentication is not configured.", {
      status: 503,
      headers: { "Cache-Control": "no-store" },
    });
  }

  const authorization = request.headers.get("authorization") ?? "";
  const encodedCredentials = authorization.match(/^Basic\s+(.+)$/i)?.[1];

  if (!encodedCredentials) {
    return authenticationRequired();
  }

  const decodedCredentials = Buffer.from(encodedCredentials, "base64").toString(
    "utf8",
  );
  const separator = decodedCredentials.indexOf(":");

  if (separator < 0) {
    return authenticationRequired();
  }

  const username = decodedCredentials.slice(0, separator);
  const password = decodedCredentials.slice(separator + 1);

  if (
    !credentialsMatch(username, expectedUsername) ||
    !credentialsMatch(password, expectedPassword)
  ) {
    return authenticationRequired();
  }

  return NextResponse.next();
}

function authenticationRequired() {
  return new Response("Authentication required.", {
    status: 401,
    headers: {
      "Cache-Control": "no-store",
      "WWW-Authenticate": 'Basic realm="Signal Desk", charset="UTF-8"',
    },
  });
}

export const config = {
  matcher: ["/((?!_next/static|_next/image|favicon.ico).*)"],
};